from openfold3.core.data.io.sequence.template import (
    A3mParser,
    parse_template_alignment,
)
import pytest
from openfold3.core.data.io.structure.cif import _load_ciffile

from biotite.database.rcsb import fetch

from openfold3.core.data.primitives.structure.metadata import (
    get_chain_to_canonical_seq_dict,
    get_cif_block,
    
)

from openfold3.core.data.io.sequence.template import (
A3mParser
)
from openfold3.core.data.primitives.structure.metadata import (
get_asym_id_to_canonical_seq_dict,
)


from pathlib import Path

class TestTemplatePreprocessor():

    def test_template_has_author_chain_id(self, tmp_path):
        """
        https://github.com/aqlaboratory/openfold-3/issues/101
        
        """

        alignment_file = Path(__file__).parent / "colabfold_template.m8"
        query_seq_str = "AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR"
        templates = parse_template_alignment(
            aln_path=Path(alignment_file),
            query_seq_str=query_seq_str,
            max_sequences=200

        )
        
        # find the offending "1rnb_A"
        template = templates[16]
        assert template.chain_id == "A" and template.entry_id == "1rnb"

        fetch(
            pdb_ids=template.entry_id,
            format="cif",
            target_path=tmp_path,
        )


        template_structure_file = tmp_path / f"{template.entry_id}.cif"

        cif_file = _load_ciffile(template_structure_file)

        chain_id_seq_map = get_asym_id_to_canonical_seq_dict(cif_file)

        cif_data = get_cif_block(cif_file)
        cif_data

        template_sequence = chain_id_seq_map.get(template.chain_id)

        parser = A3mParser(max_sequences=None)
        with pytest.raises(IndexError):
            parser(
                (
                    f">query_X/1-{len(query_seq_str)}\n"
                    f"{query_seq_str}\n"
                    f">{template.entry_id}_{template.chain_id}/{1}-{len(template_sequence)}\n"
                    f"{template_sequence}\n"
                ),
                query_seq_str,
                realign=True,
            )
            