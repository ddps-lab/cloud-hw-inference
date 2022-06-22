# save
python3 img_clf_save_model.py --model resnet50
python3 img_clf_save_model.py --model inception_v3
python3 img_clf_save_model.py --model xception
python3 img_clf_save_model.py --model vgg16
python3 img_clf_save_model.py --model mobilenet_v2

# inference for batch_list 
python3 img_clf_inference.py --model mobilenet_v2
python3 img_clf_inference.py --model resnet50
python3 img_clf_inference.py --model xception
python3 img_clf_inference.py --model inception_v3
python3 img_clf_inference.py --model vgg16
