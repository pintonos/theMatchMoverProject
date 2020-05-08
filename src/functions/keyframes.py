from functions import *
from util import *

MIN_MATCHES = 200

def find_next_key_frame(idx1, idx2):
    """
    finds point matches that are preserved between idx1 and idx2
    :param video: video file with more than |idx2| frames
    :param idx1: index to start with
    :param idx2: index to end with
    :return: list of dict with structure [{'startPoint': 2DPoint, 'endPoint': 2DPoint}],
        startPoint is the 2DPoint of the first frame, endPoint is the 2DPoint of the last frame
    """

    video, _ = get_video_streams()
    if idx2 - idx1 <= 0:
        print("warning, called find_trace_points with 0 or negative frame indexes")
        return []

    curr_idx = -1
    success = True
    keyframes = []
    traced_matches = None
    prev_frame = None
    new_keyframe_pos = 0
    keyframe_found = False
    while success and curr_idx < idx2 - idx1 - 1:
        success, frame = video.read()
        curr_idx += 1

        if curr_idx <= idx1:
            prev_frame = frame
            continue

        # trace
        match_points_1, match_points_2, matches = get_points(prev_frame, frame)

        if traced_matches is None:
            traced_matches = [{
                'start_frame': curr_idx - 1,
                'coordinates': [match_points_1[i], match_points_2[i]],
                'from': x.queryIdx,
                'to': x.trainIdx} for i, x in enumerate(matches)]
        else:
            new_matches = dict([(x.queryIdx, x.trainIdx) for x in matches])
            for match in traced_matches:
                new_from = match['to']
                if new_from in new_matches:
                    match['from'] = new_from
                    match['to'] = new_matches[new_from]
                    match['coordinates'].append(match_points_2[list(new_matches.keys()).index(new_from)])
                else:
                    match['to'] = None

            traced_matches = list(filter(lambda m: m['to'] is not None, traced_matches))

            if len(traced_matches) <= MIN_MATCHES:
                # new keyframe
                new_keyframe_pos = curr_idx + idx1 + 1
                print('found keyframe at pos ' + str(new_keyframe_pos))
                keyframes.append(traced_matches)
                keyframe_found = True
                break

        prev_frame = frame
    video.release()
    if keyframe_found:
        return keyframes, new_keyframe_pos
    else:
        print('no new keyframe found')
        return keyframes, None