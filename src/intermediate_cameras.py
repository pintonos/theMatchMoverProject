from functions import *
import cv2
import pandas as pd

MIN_MATCHES = 200


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

        if curr_idx <= idx1:
            last_frame = frame
            continue

        # trace
        match_points_1, match_points_2, matches = get_points(last_frame, frame)

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
                print('found keyframe at pos ' + str(curr_idx + 1))
                keyframes.append(traced_matches)
                traced_matches = None
        last_frame = frame

    keyframes.append(traced_matches)
    return keyframes


reader, writer = get_video_streams()
frames_total = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

pts = pd.read_csv(REF_POINTS_0, sep=',', header=None, dtype=float).values
dst = pts[:4]  # axis points from first frame

keyframes = find_key_frames(reader, 0, 100)
# comment on keyframes: each point starts in keyframe, no intermediate tracing

reader.release()
reader, _ = get_video_streams()

pts1 = []
pts2 = []
for keyframe in keyframes:
    # homography between consecutive frames
    number_intermediate_frames = len(keyframe[0]['coordinates'])

    # get points in format [[img1_kp1, img1_kp2, ...], [img2_kp1, img2_kp2, ...], ...]
    pts_list = []
    for i in range(number_intermediate_frames):
        pts = []
        for frame in keyframe:
            pts.append(frame['coordinates'][i])
        pts_list.append(pts)
    pts_list = np.asarray(pts_list)

    current_frame_idx = keyframe[0]['start_frame']
    for i in range(len(pts_list)-1):
        # get homography between each pair of frames between keyframes
        M, _ = cv2.findHomography(pts_list[i], pts_list[i+1], cv2.RANSAC, 5.0)
        dst = cv2.perspectiveTransform(dst.reshape(-1, 1, 2), M)

        # read frame at index
        current_frame_idx += 1
        _, current_frame = reader.read()

        draw_axis(current_frame, dst)
        writer.write(current_frame)


    # break after first keyframe
    # TODO resectioning
    break

reader.release()
writer.release()
