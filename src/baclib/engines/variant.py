import numpy as np

from baclib.containers.seq import SeqBatch
from baclib.core.interval import IntervalBatch
from baclib.containers.alignment import AlignmentBatch, Cigar, CigarOp
from baclib.containers.delta import MutationBatch
from baclib.lib.resources import jit


# Classes --------------------------------------------------------------------------------------------------------------
class VariantCaller:
    """
    Engine for calling variants from alignments.
    """
    @staticmethod
    def call_variants(alignments: AlignmentBatch, queries: SeqBatch, targets: SeqBatch) -> MutationBatch:
        """
        Identifies mutations (SNPs, Indels) from an AlignmentBatch.
        
        Args:
            alignments: The alignment batch.
            queries: The query sequences (SeqBatch).
            targets: The target sequences (SeqBatch).
            
        Returns:
            A MutationBatch containing all identified variants.
        """
        # We need to iterate alignments and parse CIGARs
        # This is hard to fully vectorize because CIGARs are variable length strings
        # and produce variable numbers of mutations.
        
        mut_starts = []
        mut_ends = []
        mut_strands = []
        mut_refs = []
        mut_alts = []
        
        # Access raw arrays for speed
        q_data, q_starts, _ = queries.arrays
        t_data, t_starts, _ = targets.arrays
        
        # Iterate alignments
        # Note: We assume alignments are valid and indices match the batches
        for i in range(len(alignments)):
            # Get Alignment Metadata
            q_idx = alignments.q_indices[i]
            t_idx = alignments.t_indices[i]
            
            # Global offsets in SeqBatch
            q_offset = q_starts[q_idx]
            t_offset = t_starts[t_idx]
            
            # Alignment start coords (0-based on sequence)
            q_aln_start = alignments.q_starts[i]
            t_aln_start = alignments.t_starts[i]
            t_strand = alignments.t_strands[i]
            
            cigar = alignments.cigars[i]
            if not cigar: continue
            
            # Parse CIGAR
            # We track current position in Query and Target
            curr_q = q_aln_start
            curr_t = t_aln_start
            
            ops, counts = Cigar.parse_into_arrays(cigar)
            
            for k in range(len(ops)):
                op = ops[k]
                count = counts[k]
                
                if op == CigarOp.EQ: # Match (=)
                    curr_q += count
                    curr_t += count
                elif op == CigarOp.X: # Mismatch (X)
                    # We have 'count' mismatches. 
                    # Ideally we check base-by-base to see if they are contiguous SNPs or MNP
                    # For simplicity, we treat contiguous X as one MNP block
                    
                    # Extract Ref (Target) and Alt (Query)
                    # Note: We need to handle strand if query is RC. 
                    # But usually AlignmentBatch stores coords relative to forward strand of SeqBatch entry.
                    
                    ref_slice = t_data[t_offset + curr_t : t_offset + curr_t + count]
                    alt_slice = q_data[q_offset + curr_q : q_offset + curr_q + count]
                    
                    # Create Mutation
                    # Interval is on Target (Ref)
                    mut_starts.append(curr_t)
                    mut_ends.append(curr_t + count)
                    mut_strands.append(t_strand)
                    mut_refs.append(targets.alphabet.seq_from(ref_slice))
                    mut_alts.append(queries.alphabet.seq_from(alt_slice))
                    
                    curr_q += count
                    curr_t += count
                    
                elif op == CigarOp.I: # Insertion (in Query, gap in Target)
                    alt_slice = q_data[q_offset + curr_q : q_offset + curr_q + count]
                    mut_starts.append(curr_t)
                    mut_ends.append(curr_t)
                    mut_strands.append(t_strand)
                    mut_refs.append(targets.alphabet.empty_seq())
                    mut_alts.append(queries.alphabet.seq_from(alt_slice))
                    curr_q += count
                    
                elif op == CigarOp.D: # Deletion (gap in Query, present in Target)
                    ref_slice = t_data[t_offset + curr_t : t_offset + curr_t + count]
                    mut_starts.append(curr_t)
                    mut_ends.append(curr_t + count)
                    mut_strands.append(t_strand)
                    mut_refs.append(targets.alphabet.seq_from(ref_slice))
                    mut_alts.append(queries.alphabet.empty_seq())
                    curr_t += count
                    
                elif op == CigarOp.M: # Match or Mismatch (Ambiguous)
                    # We need to compare sequences to find SNPs
                    # This is expensive in Python loop. 
                    # Ideally we use extended CIGAR (=/X) from the start.
                    # If forced to use M, we would need a kernel to scan q/t arrays.
                    curr_q += count
                    curr_t += count
                
                elif op == CigarOp.S: # Soft Clip
                    curr_q += count
                    # Soft clips consume query but not target, and are not mutations

        # Construct Batch
        if not mut_starts:
            return MutationBatch.empty()
            
        return MutationBatch(
            IntervalBatch(
                np.array(mut_starts, dtype=np.int32),
                np.array(mut_ends, dtype=np.int32),
                np.array(mut_strands, dtype=np.int32)
            ),
            targets.alphabet.batch_from(mut_refs),
            queries.alphabet.batch_from(mut_alts)
        )

    @staticmethod
    def call_pileup_variants(alignments: AlignmentBatch,
                             queries: SeqBatch, targets: SeqBatch,
                             min_depth: int = 10, min_freq: float = 0.2) -> MutationBatch:
        """
        Calls consensus variants (SNPs/Deletions) based on alignment pileup statistics.
        
        Args:
            alignments: The alignment batch.
            queries: The query sequences (SeqBatch).
            targets: The target sequences (SeqBatch).
            min_depth: Minimum coverage depth to call a variant.
            min_freq: Minimum allele frequency (count / depth) to call a variant.
            
        Returns:
            A MutationBatch of consensus variants.
        """
        mut_starts = []
        mut_ends = []
        mut_strands = []
        mut_refs = []
        mut_alts = []
        
        q_data, q_starts, _ = queries.arrays
        t_data, t_starts, t_lengths = targets.arrays
        
        # RC Table for handling reverse strand alignments
        rc_table = queries.alphabet.complement
        if rc_table is None:
            # Fallback identity if no complement (e.g. Protein), though pileup usually implies DNA
            rc_table = np.arange(len(queries.alphabet), dtype=np.uint8)

        # Process per target to keep memory usage low
        for t_idx, group_indices in alignments.group_by(by_target=True):
            t_len = t_lengths[t_idx]
            t_offset = t_starts[t_idx]
            
            # Counts: [Pos, Base]. Base 0-3 (ACGT), 4 (Deletion)
            # Assuming DNA alphabet size 4. If larger, we need to adjust.
            n_sym = len(targets.alphabet)
            counts = np.zeros((t_len, n_sym + 1), dtype=np.int32)
            
            # 1. Accumulate Counts
            for aln_i in group_indices:
                q_idx = alignments.q_indices[aln_i]
                q_offset = q_starts[q_idx]
                
                q_s = alignments.q_starts[aln_i]
                t_s = alignments.t_starts[aln_i]
                strand = alignments.t_strands[aln_i] # Relative strand
                
                cigar = alignments.cigars[aln_i]
                if not cigar: continue
                
                _pileup_add_kernel(
                    counts, q_data, q_offset, q_s, t_s, 
                    cigar, Cigar._BYTE_TO_OP,
                    strand, rc_table
                )

            # 2. Call Variants
            # We scan the counts matrix for non-ref alleles
            ref_seq_data = t_data[t_offset : t_offset + t_len]
            
            for pos in range(t_len):
                total = np.sum(counts[pos])
                if total < min_depth: continue
                
                ref_base = ref_seq_data[pos]
                
                # Check for variants
                for allele in range(n_sym + 1):
                    if allele == ref_base: continue
                    
                    count = counts[pos, allele]
                    if count == 0: continue
                    
                    freq = count / total
                    if freq >= min_freq:
                        # Found a variant!
                        mut_starts.append(pos)
                        mut_ends.append(pos + 1)
                        mut_strands.append(1)
                        
                        # Ref Seq
                        mut_refs.append(targets.alphabet.seq_from(ref_seq_data[pos:pos+1]))
                        
                        # Alt Seq
                        if allele == n_sym: # Deletion
                            mut_alts.append(queries.alphabet.empty_seq())
                        else:
                            # We need to construct a 1-byte array for the alt base
                            alt_arr = np.array([allele], dtype=np.uint8)
                            mut_alts.append(queries.alphabet.seq_from(alt_arr))

        if not mut_starts:
            return MutationBatch.empty()

        return MutationBatch(
            IntervalBatch(
                np.array(mut_starts, dtype=np.int32),
                np.array(mut_ends, dtype=np.int32),
                np.array(mut_strands, dtype=np.int32)
            ),
            targets.alphabet.batch_from(mut_refs),
            queries.alphabet.batch_from(mut_alts)
        )


# Kernels --------------------------------------------------------------------------------------------------------------
@jit(nopython=True, cache=True, nogil=True)
def _pileup_add_kernel(counts, q_data, q_offset, q_start, t_start, cigar, map_table, strand, rc_table):
    """
    Updates the pileup counts matrix based on a single alignment CIGAR.
    """
    n_cigar = len(cigar)
    curr_q = q_start
    curr_t = t_start
    
    idx = 0
    curr_count = 0
    
    # Parse CIGAR on the fly
    for i in range(n_cigar):
        b = cigar[i]
        if 48 <= b <= 57: # 0-9
            curr_count = (curr_count * 10) + (b - 48)
        else:
            op = map_table[b]
            
            # M(0), =(7), X(8) -> Consume both
            if op == 0 or op == 7 or op == 8:
                for k in range(curr_count):
                    # Get Query Base
                    base = q_data[q_offset + curr_q + k]
                    
                    # Handle Reverse Strand: If aligned to reverse, the query base 
                    # physically supports the complement on the reference forward strand.
                    if strand == -1:
                        base = rc_table[base]
                    
                    # Add to counts (relative to target start 0)
                    # counts is shape (t_len, 5)
                    # We assume caller ensures bounds, or we check:
                    if curr_t + k < len(counts):
                        counts[curr_t + k, base] += 1
                
                curr_q += curr_count
                curr_t += curr_count
                
            # I(1), S(4) -> Consume Query only
            elif op == 1 or op == 4:
                curr_q += curr_count
                # Insertions are not tracked in this simple SNP/Del pileup matrix
                
            # D(2), N(3) -> Consume Target only
            elif op == 2 or op == 3:
                del_code = counts.shape[1] - 1
                for k in range(curr_count):
                    if curr_t + k < len(counts):
                        counts[curr_t + k, del_code] += 1
                curr_t += curr_count
                
            # H(5), P(6) -> Consume neither
            
            curr_count = 0