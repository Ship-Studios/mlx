from __future__ import annotations


class _StopSequenceFilterFeedMixin:
    def feed(self, text: str):
        if not self.stops:
            return text, False
        self.buf += text
        idx, which = -1, None
        for s in self.stops:
            i = self.buf.find(s)
            if i != -1 and (idx == -1 or i < idx):
                idx, which = i, s
        if idx != -1:
            emit = self.buf[:idx]
            self.buf = ""
            self.matched = which
            return emit, True
        hold = self.max_len - 1
        if hold <= 0 or len(self.buf) <= hold:
            if hold <= 0:
                emit, self.buf = self.buf, ""
                return emit, False
            return "", False
        emit = self.buf[:-hold]
        self.buf = self.buf[-hold:]
        return emit, False
