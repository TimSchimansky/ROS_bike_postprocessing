

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

def extend_timespan_with_factor(time_span, time_multiplicator):
    """time_span is expected as tuple. Factor as number"""
    try:
        encounter_begin = (time_span[0].astype(int) / 10 ** 9)[0]
        encounter_end = (time_span[1].astype(int) / 10 ** 9)[0]
    except:
        encounter_begin, encounter_end = time_span

    # Calc length
    encounter_duration = encounter_end - encounter_begin

    # Subtract 1 from multiplicator and halve to archieve symmetrical extension value
    single_buffer_only = (time_multiplicator - 1) / 2 * encounter_duration

    # Add and subtract from start and end value
    encounter_begin -= single_buffer_only
    encounter_end += single_buffer_only

    return encounter_begin, encounter_end
