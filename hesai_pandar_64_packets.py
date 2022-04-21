# This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild

from pkg_resources import parse_version
import kaitaistruct
from kaitaistruct import KaitaiStruct, KaitaiStream, BytesIO


if parse_version(kaitaistruct.__version__) < parse_version('0.9'):
    raise Exception("Incompatible Kaitai Struct Python API: 0.9 or later is required, but you have %s" % (kaitaistruct.__version__))

class HesaiPandar64Packets(KaitaiStruct):
    def __init__(self, _io, _parent=None, _root=None):
        self._io = _io
        self._parent = _parent
        self._root = _root if _root else self
        self._read()

    def _read(self):
        self.header = HesaiPandar64Packets.Header(self._io, self, self._root)
        self.blocks = [None] * (self.header.n_blocks)
        for i in range(self.header.n_blocks):
            self.blocks[i] = HesaiPandar64Packets.Block(self._io, self, self._root)

        self.tail = HesaiPandar64Packets.Tail(self._io, self, self._root)

    class Channel(KaitaiStruct):
        def __init__(self, i, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self.i = i
            self._read()

        def _read(self):
            self.distance_value_raw = self._io.read_u2le()
            self.reflectance_value = self._io.read_u1()

        @property
        def distance_value(self):
            if hasattr(self, '_m_distance_value'):
                return self._m_distance_value if hasattr(self, '_m_distance_value') else None

            self._m_distance_value = (self.distance_value_raw * self._parent._parent.header.distance_unit)
            return self._m_distance_value if hasattr(self, '_m_distance_value') else None


    class Tail(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self._raw_dummy3 = self._io.read_bytes(5)
            _io__raw_dummy3 = KaitaiStream(BytesIO(self._raw_dummy3))
            self.dummy3 = HesaiPandar64Packets.Dummy(_io__raw_dummy3, self, self._root)
            self.high_tmp_flag = self._io.read_u1()
            self._raw_dummy4 = self._io.read_bytes(2)
            _io__raw_dummy4 = KaitaiStream(BytesIO(self._raw_dummy4))
            self.dummy4 = HesaiPandar64Packets.Dummy(_io__raw_dummy4, self, self._root)
            self.motor_speed = self._io.read_u2le()
            self.gps_timestamp = self._io.read_u4le()
            self.return_mode = self._io.read_u1()
            self.factory_info = self._io.read_u1()
            self.date_time = HesaiPandar64Packets.DateTime(self._io, self, self._root)


    class Dummy(KaitaiStruct):
        """Empty."""
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            pass


    class Block(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.azimuth_raw = self._io.read_u2le()
            self.channel = [None] * (self._parent.header.n_lasers)
            for i in range(self._parent.header.n_lasers):
                self.channel[i] = HesaiPandar64Packets.Channel(i, self._io, self, self._root)


        @property
        def azimuth_deg(self):
            if hasattr(self, '_m_azimuth_deg'):
                return self._m_azimuth_deg if hasattr(self, '_m_azimuth_deg') else None

            self._m_azimuth_deg = (self.azimuth_raw / 100)
            return self._m_azimuth_deg if hasattr(self, '_m_azimuth_deg') else None


    class DateTime(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.year_raw = self._io.read_u1()
            self.month = self._io.read_u1()
            self.day = self._io.read_u1()
            self.hour = self._io.read_u1()
            self.minute = self._io.read_u1()
            self.second = self._io.read_u1()

        @property
        def year(self):
            if hasattr(self, '_m_year'):
                return self._m_year if hasattr(self, '_m_year') else None

            self._m_year = (self.year_raw + 2000)
            return self._m_year if hasattr(self, '_m_year') else None


    class Header(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.start_of_packet = self._io.read_bytes(2)
            self.n_lasers = self._io.read_u1()
            self.n_blocks = self._io.read_u1()
            self._raw_dummy1 = self._io.read_bytes(1)
            _io__raw_dummy1 = KaitaiStream(BytesIO(self._raw_dummy1))
            self.dummy1 = HesaiPandar64Packets.Dummy(_io__raw_dummy1, self, self._root)
            self.distance_unit = self._io.read_u1()
            self._raw_dummy2 = self._io.read_bytes(2)
            _io__raw_dummy2 = KaitaiStream(BytesIO(self._raw_dummy2))
            self.dummy2 = HesaiPandar64Packets.Dummy(_io__raw_dummy2, self, self._root)


    @property
    def pandar_64_elevation_lut(self):
        if hasattr(self, '_m_pandar_64_elevation_lut'):
            return self._m_pandar_64_elevation_lut if hasattr(self, '_m_pandar_64_elevation_lut') else None

        self._m_pandar_64_elevation_lut = [14.803, 10.953, 7.98, 4.978, 2.961, 1.949, 1.781, 1.609, 1.443, 1.272, 1.105, 0.934, 0.767, 0.596, 0.429, 0.258, 0.09, -0.079, -0.248, -0.416, -0.587, -0.754, -0.924, -1.092, -1.263, -1.43, -1.601, -1.767, -1.939, -2.107, -2.277, -2.444, -2.615, -2.779, -2.952, -3.119, -3.289, -3.454, -3.627, -3.791, -3.963, -4.129, -4.3, -4.464, -4.637, -4.799, -4.971, -5.136, -5.308, -5.47, -5.644, -5.805, -5.977, -6.14, -7.142, -8.138, -9.139, -10.124, -11.111, -12.085, -13.053, -14.009, -18.968, -24.976]
        return self._m_pandar_64_elevation_lut if hasattr(self, '_m_pandar_64_elevation_lut') else None

    @property
    def pandar_64_azimuth_corr_lut(self):
        if hasattr(self, '_m_pandar_64_azimuth_corr_lut'):
            return self._m_pandar_64_azimuth_corr_lut if hasattr(self, '_m_pandar_64_azimuth_corr_lut') else None

        self._m_pandar_64_azimuth_corr_lut = [-1.042, -1.042, -1.042, -1.042, -1.042, -1.042, 1.042, 3.125, 5.208, -5.208, -3.125, -1.042, 1.042, 3.125, 5.208, -5.208, -3.125, -1.042, 1.042, 3.125, 5.208, -5.208, -3.125, -1.042, 1.042, 3.125, 5.208, -5.208, -3.125, -1.042, 1.042, 3.125, 5.208, -5.208, -3.125, -1.042, 1.042, 3.125, 5.208, -5.208, -3.125, -1.042, 1.042, 3.125, 5.208, -5.208, -3.125, -1.042, 1.042, 3.125, 5.208, -5.208, -3.125, -1.042, -1.042, -1.042, -1.042, -1.042, -1.042, -1.042, -1.042, -1.042, -1.042, -1.042]
        return self._m_pandar_64_azimuth_corr_lut if hasattr(self, '_m_pandar_64_azimuth_corr_lut') else None


