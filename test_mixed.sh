CUDA_VISIBLE_DEVICES=0 python apply_net.py --dataset-dir /home/u202432803010/datasets/coco --test-dataset coco_mixed_val --config-file VOC-Detection/faster-rcnn/Iou_FFN.yaml --inference-config Inference/standard_nms.yaml --random-seed 0 --image-corruption-level 0 --visualize 0

cd evaluator/

CUDA_VISIBLE_DEVICES=0 python eval.py --dataset-dir /home/u202432803010/datasets/coco --test-dataset coco_mixed_val --outputdir ../output/  --config-file VOC-Detection/faster-rcnn/Iou_FFN.yaml --inference-config Inference/standard_nms.yaml --random-seed 0 --image-corruption-level 0

CUDA_VISIBLE_DEVICES=0 python aose.py --dataset-dir /home/u202432803010/datasets/coco --test-dataset coco_mixed_val --outputdir ../output/  --config-file VOC-Detection/faster-rcnn/Iou_FFN.yaml --inference-config Inference/standard_nms.yaml --random-seed 0 --image-corruption-level 0

CUDA_VISIBLE_DEVICES=0 python WI.py --dataset-dir /home/u202432803010/datasets/coco --test-dataset coco_mixed_val --outputdir ../output/  --config-file VOC-Detection/faster-rcnn/Iou_FFN.yaml --inference-config Inference/standard_nms.yaml --random-seed 0 --image-corruption-level 0

# cd ..   