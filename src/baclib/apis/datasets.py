"""Client for the NCBI Datasets API, supporting genome, gene, and virus data retrieval."""
from typing import List, Optional, Union, Any, Iterator
from pathlib import Path
import os
import zipfile
import shutil
import tempfile
import json
from enum import Enum

from baclib.apis import ApiClient, Token
from baclib.io import SeqFile, Record


# Classes --------------------------------------------------------------------------------------------------------------
class GenomeTag(str, Enum):
    """File types available in an NCBI genome dataset package."""
    GENOME_FASTA = "GENOME_FASTA"
    GENOME_GFF = "GENOME_GFF"
    GENOME_GB = "GENOME_GB"
    RNA_FASTA = "RNA_FASTA"
    PROT_FASTA = "PROT_FASTA"
    CDS_FASTA = "CDS_FASTA"
    SEQUENCE_REPORT = "SEQUENCE_REPORT"


class GeneTag(str, Enum):
    """File types available in an NCBI gene dataset package."""
    GENE_FASTA = "GENE_FASTA"
    PROTEIN_FASTA = "PROTEIN_FASTA"
    CDS_FASTA = "CDS_FASTA"
    GENE_FLANKING_FASTA = "GENE_FLANKING_FASTA"


class VirusTag(str, Enum):
    """File types available in an NCBI virus dataset package."""
    GENOME = "GENOME"
    PROTEIN = "PROTEIN"
    CDS = "CDS"
    ANNOTATION = "ANNOTATION"


class NcbiToken(Token):
    """Base class for NCBI tokens."""
    pass

class GenomeAccession(NcbiToken):
    """NCBI Genome Accession (e.g. GCF_000005845.2)."""
    pass

class GeneId(NcbiToken):
    """NCBI Gene ID."""
    pass


class DatasetsClient(ApiClient):
    """
    Client for NCBI Datasets v2 API.
    """
    def __init__(self, api_key: str = None):
        super().__init__(
            base_url="https://api.ncbi.nlm.nih.gov/datasets/v2",
            api_key=api_key or os.environ.get("NCBI_API_KEY"),
            requests_per_second=10 if api_key else 3
        )

    def download_genome(self, accessions: Union[str, GenomeAccession, List[Union[str, GenomeAccession]]], 
                        output_file: Union[str, Path, None] = None,
                        include: List[Union[str, GenomeTag]] = None,
                        chromosomes: List[str] = None,
                        dehydrated: bool = False
                        ) -> 'DatasetPackage':
        """
        Download a genome dataset package.
        
        Args:
            accessions: Single accession or list of accessions (GCF_/GCA_).
            output_file: Path to save the ZIP file. If None, saves to a temporary file.
            include: List of file types to include.
            chromosomes: filter by chromosome name.
            dehydrated: If True, download a dehydrated package (fetch.txt only).
        """
        if isinstance(accessions, str):
            accessions = [accessions]
            
        endpoint = "/genome/download"
        
        # Convert enums to strings
        include_str = [str(i.value if isinstance(i, Enum) else i) for i in (include or [])]
        
        payload = {
            "accessions": accessions,
            "include_annotation_type": include_str if include_str else None,
            "chromosomes": chromosomes,
            "hydrated": "DATA_REPORT_ONLY" if dehydrated else "FULLY_HYDRATED"
        }
        
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}
        
        if output_file is None:
            # Create a named temp file that persists
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
            output_file = tf.name
            tf.close()
            
        self.download(endpoint, output_file, method='POST', json=payload)
        
        return DatasetPackage(output_file)

    def get_genome_report(self, accessions: Union[str, GenomeAccession, List[Union[str, GenomeAccession]]]) -> Any:
        """
        Get genome metadata report.
        """
        if isinstance(accessions, str):
            accessions = [accessions]
            
        endpoint = "/genome/dataset_report"
        payload = {"accessions": accessions}
        
        return self.post(endpoint, json=payload)


    def download_gene(self, gene_ids: Union[int, str, GeneId, List[Union[int, str, GeneId]]],
                      output_file: Union[str, Path, None] = None,
                      include: List[Union[str, GeneTag]] = None,
                      filename: str = None) -> 'DatasetPackage':
        """
        Download a gene dataset package.
        """
        if isinstance(gene_ids, (int, str)):
            gene_ids = [str(gene_ids)]
        else:
            gene_ids = [str(g) for g in gene_ids]
            
        endpoint = "/gene/download"
        
        include_str = [str(i.value if isinstance(i, Enum) else i) for i in (include or [])]

        payload = {
            "gene_ids": gene_ids,
            "include_annotation_type": include_str if include_str else None
        }
        # Remove None
        payload = {k: v for k, v in payload.items() if v is not None}
        
        if output_file is None:
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
            output_file = tf.name
            tf.close()

        self.download(endpoint, output_file, method='POST', json=payload)
        return DatasetPackage(output_file)

    def download_virus(self, accessions: Union[str, GenomeAccession, List[Union[str, GenomeAccession]]] = None,
                       taxon: str = None,
                       output_file: Union[str, Path, None] = None,
                       include: List[Union[str, VirusTag]] = None) -> 'DatasetPackage':
        """
        Download a virus dataset package.
        """
        if output_file is None:
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
            output_file = tf.name
            tf.close()
            
        include_str = [str(i.value if isinstance(i, Enum) else i) for i in (include or [])]

        # Virus API has multiple endpoints depending on query type
        if accessions:
            if isinstance(accessions, str): accessions = [accessions]
            endpoint = "/virus/genome/download"
            payload = {"accessions": accessions, "include_annotation_type": include_str if include_str else None}
            payload = {k:v for k,v in payload.items() if v}
            self.download(endpoint, output_file, method='POST', json=payload)
        elif taxon:
            endpoint = f"/virus/taxon/{taxon}/genome/download"
            params = {}
            if include_str: params['include_annotation_type'] = include_str
            self.download(endpoint, output_file,  method='GET', params=params)
        else:
            raise ValueError("Must provide accessions or taxon for virus download.")
            
        return DatasetPackage(output_file)


class DatasetPackage:
    """
    Helper to access files within a downloaded NCBI Dataset ZIP package.
    """
    def __init__(self, zip_path: Union[str, Path]):
        self.zip_path = Path(zip_path)
        if not self.zip_path.exists():
            # If it's a temp file created by us, it should exist.
            raise FileNotFoundError(f"Package not found: {self.zip_path}")
            
    def __repr__(self):
        return f"DatasetPackage({self.zip_path})"
        
    def extract(self, extract_path: Union[str, Path] = None):
        """Extracts the entire package."""
        with zipfile.ZipFile(self.zip_path, 'r') as z:
            z.extractall(extract_path)
            
    def iter_files(self, pattern: str = "*") -> Iterator[str]:
        """Yields filenames in the zip matching a pattern."""
        with zipfile.ZipFile(self.zip_path, 'r') as z:
            for name in z.namelist():
                if pattern == "*" or pattern in name: 
                     yield name
                     
    def sequences(self, fmt: Optional[Union[str, SeqFileFormat]] = None) -> Iterator[Record]:
        """
        Yields Record objects from files in the package that match supported sequence formats.
        
        Args:
            fmt: Optional format to filter by (e.g. SeqFileFormat.FASTA).
                 If None, tries to detect format for every file in the registry.
        """
        target_fmt = SeqFileFormat(fmt) if fmt else None
        
        with zipfile.ZipFile(self.zip_path, 'r') as z:
            for name in z.namelist():
                # Skip directories
                if name.endswith('/'): continue
                
                detected_fmt = None
                
                # Check against registry
                # If target_fmt is provided, only check that.
                # Else check all.
                
                candidates = [target_fmt] if target_fmt else SeqFile._REGISTRY.keys()
                
                for candidate in candidates:
                    spec = SeqFile._REGISTRY.get(candidate)
                    if spec and any(name.endswith(ext) for ext in spec.extensions):
                        detected_fmt = candidate
                        break
                
                if detected_fmt:
                    with z.open(name) as f:
                         # SeqFile.open with a file-like object and explicit format
                         reader = SeqFile.open(f, fmt=detected_fmt)
                         for item in reader:
                             if isinstance(item, Record):
                                 yield item

