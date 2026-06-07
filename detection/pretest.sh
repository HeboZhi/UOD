python apply_net.py --dataset-dir /home/u202432803010/datasets/voc --test-dataset voc_completely_annotation_pretest --config-file /home/u202432803010/detection/configs/VOC-Detection/faster-rcnn/Iou_FFN.yaml --inference-config /home/u202432803010/detection/configs/Inference/standard_nms.yaml --random-seed 0 --image-corruption-level 0 --visualize 0  --pretest True

cd evaluator/

python grid_traverse.py --dataset-dir /home/u202432803010/datasets/voc --test-dataset voc_completely_annotation_pretest --outputdir ../output/  --config-file VOC-Detection/faster-rcnn/Iou_FFN.yaml --inference-config /home/u202432803010/detection/configs/Inference/standard_nms.yaml --random-seed 0 --image-corruption-level 0

python energy_thresh.py  --dataset-dir /home/u202432803010/datasets/voc --config-file VOC-Detection/faster-rcnn/Iou_FFN.yaml --inference-config /home/u202432803010/detection/configs/Inference/standard_nms.yaml --image-corruption-level 0

cd ..