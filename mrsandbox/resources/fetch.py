####################################################################################################
#                                             fetch.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 05/09/26                                                                                #
#                                                                                                  #
# Purpose: Named remote data, fetched on first use into a local cache and verified against the     #
#          publisher's checksum. The package stays lightweight; large resources live at their      #
#          canonical homes and are pulled only when a simulation asks for them.                    #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import hashlib
import io
import os
import pathlib
import shutil
import urllib.request
import zipfile

from mrsandbox.resources import (bigbrain, challenge, cohorts, colin27, girf, gudmundson, hmri, icbm,
                                 mrshub, qsm,
                                 uk7t)


#*************#
#   fetched   #
#*************#
def _read(url, headers=None):
    """The bytes at `url`, tried three times (a mirror drops a request now and then)."""
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}),
                                        timeout=120) as response:
                return response.read()
        except OSError:
            if attempt == 2:
                raise


#**************************************************************************************************#
#                                           Class _Ranged                                          #
#**************************************************************************************************#
class _Ranged(io.RawIOBase):
    """A remote file read by HTTP range requests, for zipfile to take the members it needs out of
    an archive without the archive being downloaded."""

    def __init__(self, url):
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as response:
            self.url, self.size = response.geturl(), int(response.headers["Content-Length"])
        self.at = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.at

    def seek(self, offset, whence=io.SEEK_SET):
        self.at = (offset, self.at + offset, self.size + offset)[whence]
        return self.at

    def readinto(self, buffer):
        if self.at >= self.size or not len(buffer):
            return 0
        end = min(self.at + len(buffer), self.size) - 1
        data = _read(self.url, {"Range": f"bytes={self.at}-{end}"})
        buffer[:len(data)] = data
        self.at += len(data)
        return len(data)


#**************************************************************************************************#
#                                          Class Resources                                         #
#**************************************************************************************************#
#                                                                                                  #
# The registry of named remote files and the cache that serves them.                               #
#                                                                                                  #
#**************************************************************************************************#
class Resources:
    """The registry of named remote files and the cache that serves them.

    `Resources.path(name)` returns the local file for a registered resource,
    downloading it into the cache on first use and verifying the publisher's checksum.
    Data whose terms do not allow redistribution, or whose host wants every user to
    accept them, is registered with a "page" in place of a "url": it is never
    fetched, and `path` says where to download it and where to place it. Small data
    kept in this repository's data/ folder carries a "repo" path: a source checkout
    copies it from there, an installed package fetches it; the wheel ships no data.
    The cache defaults to ~/.cache/mrsandbox and can be moved with the MRSANDBOX_CACHE
    environment variable.
    """

    CACHE_ENV = "MRSANDBOX_CACHE"

    # one dict per data source, each in its own cited module
    REGISTRY = {**bigbrain.BIGBRAIN_MR, **icbm.ICBM152, **icbm.NIHPD, **colin27.COLIN27,
                **mrshub.MM_CONSENSUS, **hmri.HMRI_DEMO, **uk7t.UK7T_DEPOSIT, **girf.NWS_DEPOSIT,
                **girf.REVERSE_DEPOSIT, **challenge.MRSI_CHALLENGE, **qsm.QSM_RC2,
                **cohorts.COHORTS, **gudmundson.GUDMUNDSON}

    #************#
    #   lookup   #
    #************#
    @classmethod
    def path(cls, name):
        """Local path of a registered resource, fetching and verifying on first use."""
        try:
            entry = cls.REGISTRY[name]
        except KeyError:
            raise KeyError(f"unknown resource {name!r}; known: "
                           f"{', '.join(sorted(cls.REGISTRY))}") from None
        local = cls.cache_dir() / entry["filename"]
        if not local.exists():
            if "url" not in entry:
                raise FileNotFoundError(f"{name} is not redistributed: download it from "
                                        f"{entry['page']} and place it at {local}")
            checkout = pathlib.Path(__file__).parents[2] / entry.get("repo", "")
            source = checkout.as_uri() if "repo" in entry and checkout.is_file() else entry["url"]
            cls._fetch(source, local, entry["md5"], entry.get("member"))
        return local

    @classmethod
    def cached(cls, name):
        """True when a registered resource is already in the cache."""
        return (cls.cache_dir() / cls.REGISTRY[name]["filename"]).exists()

    @classmethod
    def cache_dir(cls):
        """The cache directory, ~/.cache/mrsandbox unless MRSANDBOX_CACHE points elsewhere."""
        root = os.environ.get(cls.CACHE_ENV, "~/.cache/mrsandbox")
        return pathlib.Path(root).expanduser()

    #***********#
    #   fetch   #
    #***********#
    @classmethod
    def _fetch(cls, url, destination, md5, member=None):
        """Download to a temporary sibling, verify the checksum (none where `md5` is None: the
        publisher states none), move into place; with `member`, that file out of the zip archive
        at `url`, read by range requests (the archive itself is never downloaded)."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".partial")
        print(f"fetching {url}{'' if member is None else '/' + member} -> {destination}")
        try:
            # a stalled connection raises after the timeout instead of hanging a job
            with (urllib.request.urlopen(url, timeout=120) if member is None
                  else cls._archive(url).open(member)) as response, open(partial, "wb") as f:
                shutil.copyfileobj(response, f)
            digest = None if md5 is None else cls._md5(partial)
            if digest != md5:
                raise IOError(f"checksum mismatch for {url}: got {digest}, expected {md5}")
            partial.replace(destination)
        finally:
            partial.unlink(missing_ok=True)

    @staticmethod
    def _archive(url):
        """The zip archive at `url` opened over HTTP range requests, for zipfile to take the
        members it needs out of it (checking each member's CRC as it reads it)."""
        return zipfile.ZipFile(io.BufferedReader(_Ranged(url), buffer_size=1 << 20))

    @staticmethod
    def _md5(path):
        digest = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
