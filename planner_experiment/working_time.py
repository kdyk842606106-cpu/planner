"""Minute-based project calendar arithmetic shared by simulator and validator."""


def next_work_start(intervals, earliest):
    if not intervals:
        return earliest
    for start,end in intervals:
        if earliest < end:
            return max(earliest,start)
    return None


def work_end(intervals, start, duration):
    if not intervals:
        return start+duration
    if next_work_start(intervals,start)!=start:
        return None
    remaining=duration
    for left,right in intervals:
        if right<=start:
            continue
        available=right-max(left,start)
        if remaining<=available:
            return max(left,start)+remaining
        remaining-=available
    return None


def work_segments(intervals, start, end):
    return [(max(a,start),min(b,end)) for a,b in intervals if max(a,start)<min(b,end)] if intervals else [(start,end)]


def next_activity_start(intervals, earliest, activity):
    windows=activity.allowed_execution_windows
    if windows is None:
        return next_work_start(intervals,earliest)
    for left,right in windows:
        start=next_work_start(intervals,max(left,earliest))
        if start is None:
            return None
        end=work_end(intervals,start,activity.duration)
        if end is not None and start<right and end<=right:
            return start
    return None


def work_offset(intervals, instant):
    """Work accumulated before an elapsed-time instant; closures have zero width."""
    return sum(max(0,min(end,instant)-start) for start,end in intervals if start<instant)


def elapsed_start(intervals, offset):
    for start,end in intervals:
        if offset < end-start:
            return start+offset
        offset-=end-start
    raise ValueError('Work start exceeds calendar horizon')


def elapsed_end(intervals, offset, origin):
    if offset==0:
        return origin
    for start,end in intervals:
        if offset <= end-start:
            return start+offset
        offset-=end-start
    raise ValueError('Work end exceeds calendar horizon')
