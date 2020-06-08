from functions import *
from util import *

MIN_MATCHES = 300


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
    while success and curr_idx < idx2:
        success, frame = video.read()
        curr_idx += 1

        if curr_idx <= idx1:
            prev_frame = frame
            continue

        # trace
        match_points_1, match_points_2, matches = get_points(prev_frame, frame, detector=Detector.ORB)

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

            if len(traced_matches) <= MIN_MATCHES or curr_idx == idx2 - 1:
                # new keyframe
                new_keyframe_pos = curr_idx + 1
                print('found keyframe at pos ' + str(new_keyframe_pos))
                keyframes.append(traced_matches)
                keyframe_found = True
                break

        prev_frame = frame
    video.release()
    if keyframe_found:
        return keyframes, new_keyframe_pos
    else:
        keyframes.append(traced_matches)
        print('last keyframe found')
        return keyframes, None


def get_all_keyframes(start_frame_idx, end_frame_idx):
    keyframes, keyframe_id = find_next_key_frame(start_frame_idx, end_frame_idx)
    keyframe_idx = [keyframe_id]
    start_idx = [start_frame_idx]
    while keyframe_id and keyframe_id < end_frame_idx:
        if len(keyframes) > 1:
            start_frame = keyframe_idx[-2]
        else:
            start_frame = start_frame_idx + len(keyframes[0][0]['coordinates']) // 2

        tmp_kf, keyframe_id = find_next_key_frame(start_frame, end_frame_idx)
        keyframes = keyframes + tmp_kf
        keyframe_idx.append(keyframe_id)
        start_idx.append(tmp_kf[0][0]['start_frame'])

    start_idx.append(end_frame_idx)
    return keyframes, start_idx


def get_keyframe_pts(keyframes):
    keyframe_pts = []
    for k in keyframes:
        pts_list = []
        for i in range(len(k[0]['coordinates'])):
            pts = []
            for frame in k:
                pts.append(frame['coordinates'][i])
            pts_list.append(pts)
        keyframe_pts.append(np.asarray(pts_list))
    return keyframe_pts
