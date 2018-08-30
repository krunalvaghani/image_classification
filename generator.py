from scipy import misc

from keras.utilsls import to_categorical


def data_generator(images,batch_size=1, taget_size=(224,224),class_mode = 'binary'):
    
    '''
    images is list of <training/validation data path> <label id>
    class_mode is 'binary' or 'multiclass'
       
    Usage in script,
    train_generator = data_generator(train_data,batch_size=2, taget_size=(224,224),class_mode = 'binary')
    val_generator = data_generator(val_data,batch_size=2, taget_size=(224,224),class_mode = 'binary')
    
    '''
    
    total_number_of_samples=len(images)
    while True:
        for offset in range(0, total_number_of_samples, batch_size):
            batch_samples = images[offset:offset + batch_size]
            X = []
            y = []
            for j in batch_samples:
                X1, y1= j.split()
                
                X1=misc.imread(in_img1)
                X1=misc.imresize(in_img1,taget_size)
                if class_mode == 'binary':
                    y=y1
                elif class_mode == 'multiclass':
                    y= to_categorical(y1)
                
                X.append(X1)
                y.append(y)
                
            X= np.asarray(X)
            y = np.asarray(y)
            
            yield X,y