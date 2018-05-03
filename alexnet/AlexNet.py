class AlexNet(object):
    def __init__(self,is_training = True):
        if is_training:
            print('Init AlexNet: Training')
        else:
            print('Init AlexNet: Inferencing...')