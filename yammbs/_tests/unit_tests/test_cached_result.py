import numpy

from yammbs.cached_result import CachedResultCollection


def test_json_roundtrip(tmp_path, small_cache):
    """Test that we can round-trip a `CachedResultCollection` through JSON."""
    with open(tmp_path / ".json", "w") as f:
        f.write(small_cache.to_json())

    new_cache = CachedResultCollection.from_json(tmp_path / ".json")

    for old, new in zip(small_cache.inner, new_cache.inner):
        assert old.mapped_smiles == new.mapped_smiles
        assert old.inchi_key == new.inchi_key
        assert old.qc_record_id == new.qc_record_id
        assert old.qc_record_final_energy == new.qc_record_final_energy
        assert numpy.allclose(old.coordinates, new.coordinates)
