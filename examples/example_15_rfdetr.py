#################################################################################
# Example of using RF-DETR detection and SORT tracking
#################################################################################

"""Run from examples/: python example_15_rfdetr.py --device cpu --frames 30.

Install requirements/rfdetr.txt from the repository root first.
Pass --checkpoint for local weights;
an empty checkpoint uses RF-DETR's download/cache. Add --show for a video window.
"""

import argparse
from pathlib import Path

import cv2

from pyppbox.ppb.mt import MT
from pyppbox.utils.visualizetools import visualizePeople


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', default=str(Path(__file__).parent / 'data/gta.mp4'))
    parser.add_argument('--model-size', choices=['nano', 'small', 'medium', 'large'], default='nano')
    parser.add_argument('--checkpoint', default='', help='Optional local checkpoint matching --model-size.')
    parser.add_argument('--person-class-id', type=int, default=1, help='COCO person ID is 1; custom models may use 0.')
    parser.add_argument('--device', default='cpu', help='cpu, cuda:0, mps, or auto')
    parser.add_argument('--frames', type=int, default=30)
    parser.add_argument('--show', action='store_true', help='Display annotated frames in an OpenCV window.')
    args = parser.parse_args()
    if args.frames < 1:
        parser.error('--frames must be positive')
    capture = cv2.VideoCapture(args.video)
    try:
        if not capture.isOpened():
            raise FileNotFoundError(f'Cannot open video: {args.video}')
        # Set a detector by dictionary and the other stages by their names.
        with MT() as pipeline:
            pipeline.setMainDetector(dict(dt_name='RF_DETR', model_size=args.model_size,
                                          model_file=args.checkpoint, person_class_id=args.person_class_id,
                                          device=args.device, conf=.4))
            pipeline.setMainTracker('SORT')
            pipeline.setMainReIDer('None')
            for index in range(args.frames):
                ok, frame = capture.read()
                if not ok:
                    break
                # Detect and track Person objects, as in the earlier examples.
                people, _ = pipeline.detectPeople(frame, img_is_mat=True)
                people = pipeline.trackPeople(frame, people, img_is_mat=True)
                print(f'Frame {index}: {len(people)} people; track IDs {[person.cid for person in people]}')
                if args.show:
                    cv2.imshow('RF-DETR + SORT', visualizePeople(frame, people))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
    finally:
        capture.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
