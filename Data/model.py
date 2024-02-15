from ultralytics import YOLO
uddoktaModel = YOLO('AI_Model/uddokta_v11.1.pt').cuda()
marchentModel = YOLO('AI_Model/marchent_v12.1.pt').cuda()

uddoktaModel.to(device=0)
marchentModel.to(device=0)
