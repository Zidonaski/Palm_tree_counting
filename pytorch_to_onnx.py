import torch
import argparse
from model import CNN
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./static/models/best_model.pt', help='weights path')
    parser.add_argument('--img-size', type=int, default=416, help='image size')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    img = torch.zeros((opt.batch_size, 3, opt.img_size,opt.img_size))
    model=CNN()
    model.load_state_dict(torch.load(opt.weights,map_location=torch.device('cpu')))
    model=model.float()
    model.eval()
    out = model(img) 
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.pt', '.onnx')  # filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['image'],
                          output_names=['output'])

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)