from functions import *
import cv2

MIN_MATCHES = 50


def find_key_frames(video, idx1, idx2):
    '''
    finds point matches that are preserved between idx1 and idx2
    :param video: video file with more than |idx2| frames
    :param idx1: index to start with
    :param idx2: index to end with
    :return: list of dict with structure [{'startPoint': 2DPoint, 'endPoint': 2DPoint}],
        startPoint is the 2DPoint of the first frame, endPoint is the 2DPoint of the last frame
    '''
    if idx2 - idx1 <= 0:
        print("warning, called find_trace_points with 0 or negative frame indexes")
        return []

    curr_idx = -1
    success = True
    keyframes = []
    traced_matches = None
    last_frame = None

    while success and curr_idx < idx2 - 1:
        success, frame = video.read()
        curr_idx += 1

        if curr_idx == idx1:
            # start tracing
            last_frame = frame

        if curr_idx > idx1:
            # trace
            match_points_1, match_points_2, matches = get_points(last_frame, frame)

            if traced_matches is None:
                traced_matches = [{
                    'startPoint': match_points_1[i],
                    'from': x.queryIdx,
                    'to': x.trainIdx,
                    'endPoint': match_points_2[i]} for i, x in enumerate(matches)]
            else:
                new_matches = dict([(x.queryIdx, x.trainIdx) for x in matches])
                for match in traced_matches:
                    new_from = match['to']
                    if new_from in new_matches:
                        match['from'] = new_from
                        match['to'] = new_matches[new_from]
                        match['endPoint'] = match_points_2[list(new_matches.keys()).index(new_from)]
                    else:
                        match['to'] = None

                traced_matches = list(filter(lambda m: m['to'] is not None, traced_matches))

                if len(traced_matches) <= MIN_MATCHES:
                    # new keyframe
                    print('found keyframe at pos ' + str(curr_idx + 1))
                    keyframes.append(traced_matches)
                    traced_matches = None

    keyframes.append(traced_matches)
    return keyframes


video = cv2.VideoCapture(VIDEO_PATH)
frames_total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

keyframes = find_key_frames(video, 0, frames_total)
print(len(keyframes))

video.release()
