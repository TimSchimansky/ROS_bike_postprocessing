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

        @property
        def z_coord(self):
            if hasattr(self, '_m_z_coord'):
                return self._m_z_coord if hasattr(self, '_m_z_coord') else None

            self._m_z_coord = (self.i * 5)
            return self._m_z_coord if hasattr(self, '_m_z_coord') else None


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
        """This type is intentionally left blank."""
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



