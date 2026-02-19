"""Client for the PRODORIC API, providing access to transcription factor binding site matrices and regulatory networks."""
from typing import List, Optional, Union, Any, Dict, Iterator
import json
import urllib.parse
from enum import Enum
import numpy as np

from baclib.apis import ApiClient
from baclib.containers.motif import Motif, Background
from baclib.containers.graph import Graph, Edge
from baclib.core.alphabet import Alphabet


# Classes --------------------------------------------------------------------------------------------------------------
class ProdoricClient(ApiClient):
    """
    Client for the PRODORIC API.
    Provides access to transcription factor binding sites and matrices.
    """
    def __init__(self, api_key: str = None):
        super().__init__(
            base_url="https://www.prodoric.de/api",
            api_key=None, 
            requests_per_second=3
        )

    def search_motifs(self, term: str) -> List[Dict[str, Any]]:
        """
        Search for matrices by term (accession, name, organism).
        Returns a list of MatrixSearchResult objects (dicts).
        """
        encoded_term = urllib.parse.quote(term)
        endpoint = f"matrix/search/{encoded_term}"
        
        try:
            return self.get(endpoint)
        except Exception:
            return []

    def get_motif(self, accession: str) -> Optional[Motif]:
        """
        Retrieve a matrix by accession (e.g. MX000001) and return a Motif object.
        """
        clean_id = accession
        if accession.startswith("MX"):
            clean_id = accession[2:]
        
        # Validation: check if clean_id is integer-like
        if not clean_id.isdigit():
             raise ValueError(f"Invalid accession format: {accession}. Expected MX followed by integers.")

        endpoint = f"matrix/MX{clean_id}"
        
        try:
            data = self.get(endpoint)
            # data could be None if 404
            if not data: return None
            return self._parse_motif(data, accession)
        except Exception:
            # We might want to log or re-raise, but for now return None on failure
            return None

    def get_regulon(self, matrix_id: str) -> Optional[Graph]:
        """
        Retrieve the regulon network for a specific matrix (TF).
        Returns a generic Graph object.
        """
        clean_id = matrix_id
        if matrix_id.startswith("MX"):
            clean_id = matrix_id[2:]
            
        endpoint = f"prodonet/MX{clean_id}"
        return self._fetch_graph(endpoint)

    def get_network(self, organism_acc: str) -> Optional[Graph]:
        """
        Retrieve the regulatory network for an organism.
        """
        endpoint = f"prodonet/{organism_acc}"
        return self._fetch_graph(endpoint)

    def _fetch_graph(self, endpoint: str) -> Optional[Graph]:
        try:
            data = self.get(endpoint)
            if not data: return None
            return self._parse_network(data)
        except Exception:
            return None

    def _parse_motif(self, data: Dict[str, Any], accession: str) -> Motif:
        """
        Convert PRODORIC matrix JSON to baclib Motif.
        """
        # Data contains "pwm" key with "A", "C", "G", "T" lists.
        # Although called "pwm", the values look like counts (integers like 0, 2, 0.5?)
        # 2 and 0.5? Maybe weighted counts?
        # Let's assume they are frequencies or weights that need to be normalized.
        # But wait, Motif.from_counts expects integers conventionally, but floats work nicely too.
        # If they sum to significantly more than 1 per column, they are counts/weights.
        # If they sum to 1, they are probabilities.
        
        pwm = data.get("pwm", {})
        if not pwm:
             raise ValueError("No PWM data found in response")

        # Order: A, C, G, T? BacLib DNA default is often lexicographical or specific.
        # Alphabet.DNA default order is typically A, C, G, T or similar.
        # Let's check BacLib alphabet.
        # Alphabet.DNA is usually A, C, G, T.
        # We need to constructing a matrix of shape (4, L).
        
        # Keys in JSON: "A", "C", "G", "T"
        # We need to map these to the Alphabet indices.
        
        # Alphabet.DNA is defined as b'TCAG' (Lines 511 in core/alphabet.py)
        # We must construct the matrix with rows in this order: T, C, A, G
        
        alphabet = Alphabet.DNA
        
        # Check length from 'A' (or any key present)
        if 'A' in pwm:
             length = len(pwm['A'])
        elif 'T' in pwm:
             length = len(pwm['T'])
        else:
             raise ValueError("PWM must contain at least 'A' or 'T'")
             
        if length == 0: raise ValueError("Empty matrix")
        
        # Explicit construction in TCAG order:
        # Row 0: T
        # Row 1: C
        # Row 2: A
        # Row 3: G
        zeros = [0.0] * length
        rows = [
            pwm.get('T', zeros),
            pwm.get('C', zeros),
            pwm.get('A', zeros),
            pwm.get('G', zeros)
        ]
        
        matrix = np.array(rows, dtype=np.float32)
        
        # The data in example has 0, 2, 0.5.
        # If we treat them as counts/weights, we can use from_counts 
        # (which normalizes to freqs) or from_weights directly?
        # from_counts does: freqs = counts / sum; then from_freqs
        # from_freqs does: weights = freqs / bg; then from_weights
        
        # If the input IS weights (odds), using from_counts is wrong.
        # If the input IS counts, using from_weights is wrong.
        # 0.5 looks like a count (maybe 0.5 sequences? weighted sequence?).
        # "max_score": 24 implies a scoring range.
        
        # Let's blindly try from_counts as a safe default for now unless we know it's log-odds.
        # (Raw PSSM values would be negative and positive floats).
        # These are all non-negative. Likely counts or frequencies.
        
        bs_name = data.get("name", "Unknown").encode('ascii', errors='ignore')
        
        bg = Background.uniform(alphabet)
        return Motif.from_counts(bs_name, matrix, bg)

    def _parse_network(self, data: Dict[str, Any]) -> Graph:
        """
        Convert ProdoNet JSON to baclib Graph.
        """
        g = Graph(directed=True, multi=True)
        
        # 1. Add Nodes
        nodes = data.get("nodes", [])
        # Each node has "id" (int), "label", "type", "mx" (optional), "group"
        # We should use "id" as the internal ID (converted to bytes) for the Graph structure?
        # Or should we use labels? IDs are safer.
        # Edge "from" and "to" use these integer IDs.
        
        node_map = {} # int id -> bytes id
        
        for n in nodes:
            nid = n.get("id")
            label = n.get("label", str(nid))
            # Create a unique bytes ID. Maybe just str(nid)?
            bid = str(nid).encode('ascii')
            node_map[nid] = bid
            
            # Attributes
            attrs = {
                b'label': label,
                b'type': n.get('type'),
                b'group': n.get('group'),
                b'mx': n.get('mx')
            }
            g.add_node(bid, attributes=attrs)
            
        # 2. Add Edges
        edges = data.get("edges", [])
        for e in edges:
            u_int = e.get("from")
            v_int = e.get("to")
            etype = e.get("type") # "-", "op", "+" ?, "repression"?
            
            if u_int in node_map and v_int in node_map:
                u = node_map[u_int]
                v = node_map[v_int]
                
                attrs = {}
                if etype: attrs[b'type'] = etype
                
                # Create Edge object
                # Default strands to forward as graph is abstract regulation
                edge = Edge(u, v, attributes=attrs)
                g.add_edges([edge])
                
        return g
