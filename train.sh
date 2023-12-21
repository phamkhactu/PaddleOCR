# recommended paddle.__version__ == 2.0.0
python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1,2,3,4,5,6,7'  tools/train.py -c configs/rec/rec_mv3_none_bilstm_ctc.yml
# python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0' tools/train.py -c configs/rec/rec_vn_r34_vd_none_bilstm_ctc.yml -o Global.pretrained_model=pretrain_models/rec_r34_vd_none_bilstm_ctc_v2.0_train/best_accuracy
# python  tools/infer_rec.py -c configs/rec/rec_vn_r34_vd_none_bilstm_ctc.yml -o Global.pretrained_model=output/rec/r34_vn_vd_none_bilstm_ctc/best_accuracy Global.load_static_weights=false Global.infer_img=doc/imgs_words/10256.jpg 
# python  tools/export_model.py -c configs/rec/rec_vn_r34_vd_none_bilstm_ctc.yml -o Global.pretrained_model=output/rec/r34_vn_vd_none_bis
#  python  tools/infer/predict_rec.py --image_dir="doc/imgs_words/0.jpg" --rec_model_dir="inference/rec_crnn_vn/" --rec_image_shape="3, 32, 150" --rec_char_dict_path="ppocr/utils/dict/vn_dict.txt"


python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0' tools/train.py -c configs/rec/PP-OCRv3/vn_PP-OCRv3_rec.yml -o Global.pretrained_model=output/rec/v3_vn_resnet34/best_accuracy
python  tools/infer_rec.py -c configs/rec/PP-OCRv3/vn_PP-OCRv3_rec.yml -o Global.pretrained_model=output/rec/v3_vn_resnet34/best_accuracy Global.load_static_weights=false Global.infer_img=doc/imgs_words/10256.jpg 
python tools/export_model.py -c configs/rec/PP-OCRv3/vn_PP-OCRv3_rec.yml -o Global.pretrained_model=output/rec/v3_vn_resnet34/best_accuracy Global.save_inference_dir=inference/rec_svtr_vn/
python  tools/infer/predict_rec.py --image_dir="doc/imgs_words/0.jpg" --rec_model_dir="inference/rec_svtr_vn/" --rec_image_shape="3, 32, 150" --rec_char_dict_path="ppocr/utils/dict/vn_dict.txt"

paddle2onnx --model_dir ./inference/rec_svtr_vn \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file ./inference/rec_onnx/svtr_r34_model.onnx \
--opset_version 10 \
--input_shape_dict="{'x':[-1,3,-1,-1]}" \
--enable_onnx_checker True