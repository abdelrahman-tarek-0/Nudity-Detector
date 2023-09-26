from nudenet import NudeDetector

import cv2
import os


def detector(path, meta={
    'type': "image",
    'intensity': 0.5,
    'interval': 10,  # Only for video
    'keep_buffer': False,  # Only for video
    'exclude' : []
}):
    nude_detector = NudeDetector()
    ignore_class = ['FACE_FEMALE', 'FEMALE_GENITALIA_COVERED', 'FEET_EXPOSED', 'BELLY_COVERED', 'FEET_COVERED', 'ARMPITS_COVERED',
                    'ARMPITS_EXPOSED', 'FACE_MALE', 'BELLY_EXPOSED', 'ANUS_COVERED', 'BUTTOCKS_COVERED', 'FEMALE_BREAST_COVERED']

    def analyze_result(result, intensity=0.5):
        if len(result) == 0:
            return {'message': 'No nudity detected', 'is_nudity': False}

        for res in result:
            class_result = res['class']
            score = res['score']

            if res['class'] in ignore_class:
                return {'message': 'No nudity detected', 'is_nudity': False}

            if score > intensity:
                return {'message': f"Detected nudity with {class_result} and score {score}", 'is_nudity': True, 'class': class_result, 'score': score}
            else:
                return {'message': 'No nudity detected', 'is_nudity': False}

    if meta['type'] == "image":
        result = nude_detector.detect(path)
        return analyze_result(result, meta['intensity'])
    
    if meta['type'] == "test":
        result = nude_detector.detect(path)
        return result

    if meta['type'] == "video":
        try:
            if not os.path.exists('buffer'):
                os.makedirs('buffer')
        except OSError:
            print('Error: Creating directory of buffer')

        currentframe = 0
        cam = cv2.VideoCapture(path)
        length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = meta['interval']
        images = []
        frames_results = []  # [{frame: 1, result: {}}]

        if length >= interval:
            frame_interval = length // interval
        else:
            frame_interval = 1

        while True:
            ret, frame = cam.read()
            if ret:
                if currentframe % frame_interval == 0:
                    name = './buffer/frame' + str(currentframe) + '.jpg'
                    cv2.imwrite(name, frame)
                    images.append(name)
                currentframe += 1
            else:
                break

        nudity_frames_count = 0
        for image in images:
            result = nude_detector.detect(image)
            analysis = analyze_result(result, meta['intensity'])
            if analysis['is_nudity']:
                nudity_frames_count += 1

            if 'frames' not in meta['exclude']:
                frames_results.append({'frame': image, 'result': analysis})

        probability = nudity_frames_count / len(images) * 100
        
        if not meta['keep_buffer']:
            os.system('rm -rf buffer')

        return {
            'message': f"Detected nudity with {probability}% probability",
            'is_nudity': probability > 50,
            'probability': probability,
            'frames': frames_results,
            'total_frames': len(images),
        }

# print(detector('buffer/frame16005.jpg', {
#     'type': "test",

# }))

test = detector('INTENSE.mp4', {
    'type': "video",
    'intensity': 0.1,
    'interval': 5,
    'keep_buffer': False,
    'exclude' : ['frames']
})

print(
    test['message'],'\n',
    test['is_nudity'],'\n',
    test['probability'],'\n',
    test['total_frames'],'\n',
)

for frame in test['frames']:
    print(frame['frame'], frame['result']['message'])

