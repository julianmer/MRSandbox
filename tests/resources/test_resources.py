####################################################################################################
#                                        test_resources.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 05/09/26                                                                                #
#                                                                                                  #
# Purpose: Resource cache gate: the registry is well formed, fetching verifies checksums and       #
#          cleans up after a mismatch, the cache is hit on the second request, and the cache       #
#          location honours the environment override. No network involved: fetches run against     #
#          file:// URLs.                                                                           #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import hashlib
import pathlib
import re
import zipfile

import pytest

from mrsandbox.resources import Resources


#*************#
#   helpers   #
#*************#
@pytest.fixture()
def local_resource(tmp_path, monkeypatch):
    """A registered resource served from a local file:// URL into a tmp cache."""
    source = tmp_path / "payload.bin"
    source.write_bytes(b"mrsandbox test payload")
    md5 = hashlib.md5(source.read_bytes()).hexdigest()
    monkeypatch.setenv(Resources.CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setitem(
        Resources.REGISTRY, "test/payload",
        {"url": source.as_uri(), "md5": md5, "filename": "test/payload.bin"},
    )
    return source, md5


#**************************************************************************************************#
#                                        Class TestRegistry                                        #
#**************************************************************************************************#
class TestRegistry:

    #*****************#
    #   well formed   #
    #*****************#
    def test_entries_are_well_formed(self):
        for name, entry in Resources.REGISTRY.items():
            assert entry["filename"], name
            if "url" not in entry:  # placed by the user, never fetched
                assert entry["page"].startswith("https://"), name
                continue
            assert entry["url"].startswith("https://"), name
            assert re.fullmatch(r"[0-9a-f]{32}", entry["md5"]), name
            if "repo" in entry:  # kept in this repository's data/ folder, never in the wheel
                source = pathlib.Path(__file__).parents[2] / entry["repo"]
                assert (entry["repo"].startswith("data/")
                        and entry["url"].endswith(entry["repo"])), name
                assert Resources._md5(source) == entry["md5"], name

    def test_unknown_name_is_a_clear_error(self):
        with pytest.raises(KeyError, match="unknown resource"):
            Resources.path("nope/nothing")


#**************************************************************************************************#
#                                         Class TestFetch                                          #
#**************************************************************************************************#
class TestFetch:

    #*****************#
    #   round trip    #
    #*****************#
    def test_fetch_verifies_and_caches(self, local_resource, monkeypatch):
        source, _ = local_resource
        path = Resources.path("test/payload")
        assert path.read_bytes() == source.read_bytes()
        assert Resources.cached("test/payload")
        # second request must come from the cache, not the url
        source.unlink()
        assert Resources.path("test/payload") == path

    #**********************#
    #   checksum failure   #
    #**********************#
    def test_checksum_mismatch_raises_and_cleans_up(self, local_resource, monkeypatch):
        monkeypatch.setitem(
            Resources.REGISTRY, "test/payload",
            {**Resources.REGISTRY["test/payload"], "md5": "0" * 32},
        )
        with pytest.raises(IOError, match="checksum mismatch"):
            Resources.path("test/payload")
        assert not Resources.cached("test/payload")

    #*******************#
    #   cache override  #
    #*******************#
    def test_cache_honours_the_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv(Resources.CACHE_ENV, str(tmp_path / "elsewhere"))
        assert Resources.cache_dir() == tmp_path / "elsewhere"

    #*******************#
    #   placed by hand  #
    #*******************#
    def test_a_member_of_an_archive_is_taken_out_by_itself(self, tmp_path, monkeypatch):
        monkeypatch.setenv(Resources.CACHE_ENV, str(tmp_path / "cache"))
        archive = tmp_path / "bundle.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("other.bin", b"not this one")
            bundle.writestr("maps/wanted.bin", b"the member's bytes")
        # the archive opened locally in place of the ranged reader over HTTP
        monkeypatch.setattr(Resources, "_archive", staticmethod(lambda url: zipfile.ZipFile(archive)))
        monkeypatch.setitem(Resources.REGISTRY, "test/member", {
            "url": "https://example.org/bundle.zip", "member": "maps/wanted.bin",
            "md5": hashlib.md5(b"the member's bytes").hexdigest(), "filename": "test/wanted.bin"})
        assert Resources.path("test/member").read_bytes() == b"the member's bytes"
        assert Resources.cached("test/member")

    def test_a_page_resource_is_never_fetched_and_says_where_to_put_it(self, tmp_path, monkeypatch):
        monkeypatch.setenv(Resources.CACHE_ENV, str(tmp_path))
        monkeypatch.setitem(Resources.REGISTRY, "test/placed",
                            {"page": "https://example.org/terms", "filename": "test/placed.nii.gz"})
        with pytest.raises(FileNotFoundError) as error:
            Resources.path("test/placed")
        assert "https://example.org/terms" in str(error.value)
        assert str(tmp_path / "test" / "placed.nii.gz") in str(error.value)
        (tmp_path / "test").mkdir()
        (tmp_path / "test" / "placed.nii.gz").write_bytes(b"data")
        assert Resources.path("test/placed") == tmp_path / "test" / "placed.nii.gz"
