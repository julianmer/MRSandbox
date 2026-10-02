####################################################################################################
#                                         test_philips.py                                          #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 19/09/26                                                                                #
#                                                                                                  #
# Purpose: Gates of io.philips on files written here in the three formats' layouts: data/list,     #
#          sdat/spar, and lab/raw/sin. Checked against spec2nii and loadLABRAW on real files by    #
#          hand, which a test cannot ship.                                                         #
#                                                                                                  #
####################################################################################################

#*************#
#   imports   #
#*************#
import numpy as np

from mrsandbox.io.philips import DataList, LabRaw, SdatSpar


#*****************#
#   data / list   #
#*****************#
def test_data_list_keeps_channels_transients_signs_and_noise_apart(tmp_path):
    rng = np.random.default_rng(0)
    n, channels, transients = 16, 3, 4
    vectors = rng.standard_normal((channels + channels * transients, n, 2)).astype("<f4")
    vectors.tofile(tmp_path / "raw_001.data")
    lines = ["# Scan name   : test", ".    0    0    0  number_of_mixes                    :     1"]
    rows = [("NOI", c, 0, 1) for c in range(channels)] + [("STD", c, t, (-1) ** t)
                                                           for t in range(transients)
                                                           for c in range(channels)]
    for k, (typ, chan, aver, sign) in enumerate(rows):
        fields = [0, 0, 0, 0, 0, chan, 0, 0, 0, 0, 0, aver, sign, 0, 0, 0, 0, 0, n * 8, k * n * 8]
        lines.append("  " + typ + " " + " ".join(str(f) for f in fields))
    (tmp_path / "raw_001.list").write_text("\n".join(lines) + "\n")

    scan = DataList.load(tmp_path / "raw_001.list")
    assert scan.parameters["Scan name"] == "test" and scan.parameters["number_of_mixes"] == "1"
    stored = vectors[..., 0] + 1j * vectors[..., 1]
    assert np.array_equal(scan.vectors("NOI")[:, :, 0], stored[:channels].T)
    data = scan.vectors("STD")
    assert data.shape == (n, channels, transients)
    assert np.array_equal(data[:, 1, 3], -stored[channels + 3 * channels + 1])
    assert np.array_equal(scan.vectors("STD", signed=False)[:, 1, 3],
                          stored[channels + 3 * channels + 1])


#*****************#
#   sdat / spar   #
#*****************#
def test_sdat_reads_vax_floats(tmp_path):
    # 1.0 and -0.75 as VAX F floats (excess-128 exponent on a 0.1f mantissa, 16-bit halves
    # swapped)
    def vax(value):
        mantissa, exponent = np.frexp(abs(value))
        word = ((int(value < 0) << 31) | ((int(exponent) + 128) << 23)
                | (int((mantissa - 0.5) * 2 ** 24) & 0x7FFFFF))
        return ((word << 16) | (word >> 16)) & 0xFFFFFFFF
    np.array([vax(1.0), vax(-0.75), vax(0.15625), 0], dtype="<u4").tofile(tmp_path / "scan.SDAT")
    (tmp_path / "scan.SPAR").write_text("!comment : no\nsamples : 2\nrows : 1\n"
                                        "sample_frequency : 2000\n")
    fids, parameters = SdatSpar.load(tmp_path / "scan.SDAT")
    assert (np.allclose(fids, [[1.0 - 0.75j, 0.15625 + 0.0j]])
            and parameters["sample_frequency"] == "2000")


#*******************#
#   lab / raw / sin #
#*******************#
def test_lab_raw_reads_the_older_plain_and_the_newer_bit_packed_stream(tmp_path):
    rng = np.random.default_rng(1)
    channels, samples = 2, 24
    blocks = rng.integers(-1000, 1000, (3, channels, samples, 2))   # noise, two normal
    labels = np.zeros((3, 16), dtype="<u4")
    labels[:, 3] = LabRaw.STANDARD_LABEL << 16
    labels[0, 4] = LabRaw.NOISE_DATA
    labels[2, 4] = (2 << 16) | (1 << 24)                            # half a turn, reversed
    stored = blocks[..., 0] + 1j * blocks[..., 1]

    # the older generation: int16 as they are, the .sin file names enable_pda
    labels[:, 0] = channels * samples * 4
    labels.tofile(tmp_path / "old.lab")
    (tmp_path / "old.sin").write_text(" 01 00 00: nr_measured_channels  :  2\n"
                                      " 01 00 00: enable_pda  :  1\n")
    with open(tmp_path / "old.raw", "wb") as handle:
        handle.write(bytes(512) + blocks.astype("<i2").tobytes())

    # the newer one: int32, bit-packed in chunks of groups of sixteen, each group its own width
    coded = []
    for block in blocks:
        values, stream = block.ravel(), b""
        for first in range(0, values.size, 40):                     # chunks of 40 values
            chunk, bits = values[first:first + 40], ""
            for g in range(0, chunk.size, 16):
                group = chunk[g:g + 16]
                width = int(max(abs(int(v)) for v in group)).bit_length() + 1
                bits += format(width, "05b") + format(0, "05b") + "".join(
                    format(int(v) % (1 << width), f"0{width}b") for v in group)
            bits += "0" * (-len(bits) % 32)
            words = np.array([int(bits[k:k + 32], 2) for k in range(0, len(bits), 32)],
                             dtype="<u4").tobytes()
            stream += (np.array([chunk.size * 4, len(words)], dtype="<u2").tobytes()
                       + np.array([first * 4], dtype="<u4").tobytes() + words)
        coded.append(stream)
    labels[:, 0] = channels * samples * 8
    labels[:, 1] = [len(stream) for stream in coded]
    labels[:, 5] = 6 << 8                                           # raw format 6
    labels.tofile(tmp_path / "new.lab")
    (tmp_path / "new.sin").write_text(" 01 00 00: nr_measured_channels  :  2\n")
    with open(tmp_path / "new.raw", "wb") as handle:
        handle.write(bytes(512) + b"".join(coded))

    for name, packed in (("old", False), ("new", True)):
        scan = LabRaw.load(tmp_path / f"{name}.raw")
        assert scan.packed is packed and scan.channels == channels
        assert np.allclose(scan.noise(), stored[0].T)
        data = scan.data()
        assert data.shape == (samples, channels, 2) and np.allclose(data[:, :, 0], stored[1].T)
        assert np.allclose(data[:, :, 1], -stored[2].T[::-1], atol=1e-3)


def test_lab_raw_comes_to_the_spectral_band_as_nifti_mrs(tmp_path):
    # eightfold samples holding a tone inside the output band and one outside it: the band keeps the
    # first at its amplitude and drops the second; the coils and the transients stay apart
    channels, points, fold = 2, 8, 8
    t = np.arange(points * fold)
    inside, outside = np.exp(2j * np.pi * 2 * t / t.size), np.exp(2j * np.pi * 20 * t / t.size)
    tone = 1000.0 * inside + 500.0 * outside
    blocks = np.stack([np.stack([np.stack([tone.real, tone.imag], -1)] * channels)] * 3)
    labels = np.zeros((3, 16), dtype="<u4")
    labels[:, 3] = LabRaw.STANDARD_LABEL << 16
    labels[0, 4] = LabRaw.NOISE_DATA
    labels[:, 0] = channels * points * fold * 4
    labels.tofile(tmp_path / "run.lab")
    (tmp_path / "run.sin").write_text(
        " 01 00 00: nr_measured_channels  :  2\n 01 00 00: enable_pda  :  1\n"
        " 01 00 00: output_resolutions  :  8  1  1  1\n 00 00 00: sample_time_interval  :  62.5\n"
        " 00 01 00: spectro_echo_times  :  35.0\n 00 01 00: spectro_rep_times  :  2000.0\n")
    with open(tmp_path / "run.raw", "wb") as handle:
        handle.write(bytes(512) + np.round(blocks).astype("<i2").tobytes())
    run = LabRaw.load(tmp_path / "run.raw")
    image = run.nifti(127.7)
    fids = np.asarray(image[:])[0, 0, 0]
    assert fids.shape == (points, channels, 2) and image.dim_tags[:2] == ["DIM_COIL", "DIM_DYN"]
    assert abs(image.dwelltime - 62.5e-6 * fold) < 1e-12
    assert image.hdr_ext["EchoTime"] == 0.035 and image.hdr_ext["RepetitionTime"] == 2.0
    assert np.allclose(fids[:, 0, 0], 1000.0 * inside[::fold], atol=1.0)
    assert run.band_noise().shape == (points, channels)
