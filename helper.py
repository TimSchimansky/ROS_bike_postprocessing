

def vec3_to_list(vector3_in):
    return [vector3_in.x, vector3_in.y, vector3_in.z]

def quaternion_to_list(quaternion_in):
    return [quaternion_in.x, quaternion_in.y, quaternion_in.z, quaternion_in.w]

class PLYWriter:
    """Reused from IKG-Intersection project 2021"""

    def __init__(self, file, fields, comments=None):
        assert(file.mode == "wb")
        self._file = file
        self._counter = 0
        header = 'ply\nformat binary_little_endian 1.0\nelement vertex 000000000000'
        if comments is not None:
            if isinstance(comments, str):
                comments = comments.split("\n")
            for comment in comments:
                header += "\ncomment {}".format(comment)
        for name, dtype in fields:
            header += "\nproperty {} {}".format(dtype.__name__, name)
        header += "\nend_header\n"
        self._file.write(bytes(header, encoding='utf-8'))

    def writeArray(self, data):
        self._file.write(data.tobytes())
        self._counter += data.shape[0]

    def writeDataFrame(self, data):
        self._file.write(data.to_records(index=False).tobytes())
        self._counter += len(data)

    def finish(self):
        self._file.seek(51)
        self._file.write(bytes("{:012d}".format(self._counter), encoding="utf-8"))

def dec_2_dms(decimal):
    minute, second = divmod(decimal*3600, 60)
    degree, minute = divmod(minute, 60)
    return '%d° %d\' %.2f\"' %(degree, minute, second)