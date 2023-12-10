class SourceDB:
    def __init__(self, sname, count=0):
        self.sname = sname
        self.count = count

    def to_dict(self):
        return {'sname': self.sname, 'count': self.count}

    @staticmethod
    def from_dict(d):
        return SourceDB(d['sname'], d['count'])
