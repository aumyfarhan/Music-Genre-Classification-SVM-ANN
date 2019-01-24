
import tensorflow as tf
import librosa
import numpy as np
import csv
import os
from sklearn.model_selection import train_test_split
from fnmatch import fnmatch

print ('start')

# read train data files

# training data directory
train_root = 'E:\ML_PROJECT3\genres'

# data type extension
pattern = "*.au"

sdir_no=0
class_dict={}
file_list_label=[]

# walk through training directory and read all .au file names
for path, subdirs, files in os.walk(train_root):
    if (len(subdirs)>1):
        for sd in subdirs:
            
            # create a dictionary of genre folder name
            # and numeric value label
            
            # give each genre a unique numeric value.
            # we start from zero, and
            # each time we encounter a new genre folder name
            # we increase the label counter by one.
            # and give the corresponding genre the current label

            
            if sd not in class_dict:
                class_dict[sd]=sdir_no
                sdir_no += 1
    
    cur_dir=os.path.basename(path)
    cur_label=class_dict.get(cur_dir," ")
    
    for name in files:
        if fnmatch(name, pattern):
            
            filename=os.path.join(path, name)
            
            cur_tuple=(filename,cur_label)
            
            # append training filenames and the corresponding
            # numeric label in list
            file_list_label.append(cur_tuple)
            

# inverse the genre name vs numeric label
# and store in a dictionary
            
inverse_CC={}        
for key,value in class_dict.items():
    inverse_CC[value]=key
    
label_list=[]
for key,value in class_dict.items():
    label_list.append(key)
    
    
# validation data directory
validation_root = 'E:\ML_PROJECT3\AA'

#data type extension
pattern = "*.au"

validation_file_list=[]

# walk through validation directory and read all .au file name
for path, subdirs, files in os.walk(validation_root):
    for name in files:
        if fnmatch(name, pattern):
            
            filename=os.path.join(path, name)
            
            # append validation filenames in list
            validation_file_list.append(filename)
            
            
# create an empty list to append all training file labels
labels_ALL2 = []

# iterate through filename list and read file data
# from file data extract MFCC feature
w=0

X_test2 = []
        
for name_label in file_list_label:
    
    cur_label=name_label[1]

    XM,SR=librosa.core.load(name_label[0])
    XM=XM[int(0.15*len(XM)):int(0.85*len(XM))]
    XM_frame=librosa.util.frame(XM, frame_length=int(SR*3)\
                              , hop_length=int(SR*3))



    for i in range(XM_frame.shape[1]):
        
        print(w)
        kk=XM_frame[:,i]
        
        mfcc_cc = librosa.core.amplitude_to_db(librosa.feature.melspectrogram(y=kk, sr=SR,n_mels=128,\
                                                 fmax=8000,hop_length=int(SR*20/1000)),ref=1)
        w=w+1
        
        mfcc_bb=mfcc_cc[:,:128]
        
        X_test2.append(mfcc_bb)

        labels_ALL2.append(cur_label)
        

X_test2=np.array(X_test2)

XX_new=[]
for i in range(X_test2.shape[0]):
    
    print(i)
    print('\n')
    cur_image=X_test2[i,:,:]
    mval=np.mean(cur_image)
    sval=np.std(cur_image)
    new_image=(cur_image-mval)/(sval+0.00000001)
    
    minval=np.min(new_image)
    maxval=np.max(new_image)
    
    new_image2=(new_image-minval)/(maxval-minval)
    
    XX_new.append(new_image2)
    
XX_new=np.array(XX_new)  
#########



def get_batches(X, y, batch_size):
    """ Return a generator for batches """
    n_batches = len(X) // batch_size
    X, y = X[:n_batches*batch_size], y[:n_batches*batch_size]

    # Loop over batches and yield
    for b in range(0, len(X), batch_size):
        yield X[b:b+batch_size], y[b:b+batch_size]
        
def get_batches_X_ALL(X,batch_size_ALL=1):
    """ Return a generator for batches """
    n_batches = len(X) // batch_size_ALL
    X= X[:n_batches*batch_size_ALL]

    # Loop over batches and yield
    for b in range(0, len(X), batch_size_ALL):
        yield X[b:b+batch_size_ALL]



labels_ALL=np.array(labels_ALL2)
X_test=XX_new.reshape(-1,128,128,1)
X_tr, X_vld, lab_tr, lab_vld = train_test_split(X_test, labels_ALL, 
                                                stratify = labels_ALL, random_state = 123)



from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()

y_tr = encoder.fit_transform(lab_tr.reshape(-1,1))

y_vld = encoder.fit_transform(lab_vld.reshape(-1,1))
#
#y_test = encoder.fit_transform(labels_test.reshape(-1,1))
#
#y_test=y_test.toarray()


batch_size = 512
batch_size_V=100     # Batch size
learning_rate = 0.001
epochs = 1000
keep_prob_=0.6
n_classes = 10
n_channels = 1

graph = tf.Graph()

# Construct placeholders
with graph.as_default():
    inputs_ = tf.placeholder(tf.float32, [None, 128,128,1], name = 'inputs')
    labels_ = tf.placeholder(tf.float32, [None, n_classes], name = 'labels')
    keep_prob_ = tf.placeholder(tf.float32, name = 'keep')
    learning_rate_ = tf.placeholder(tf.float32, name = 'learning_rate')

with graph.as_default():
    
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.0019)
    
    conv1 = tf.layers.conv2d(inputs=inputs_, filters=64, kernel_size=[2,2], strides=2, 
                             padding='same', activation = tf.nn.relu\
                             ,kernel_regularizer=regularizer)
    max_pool_1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=1, padding='same')

    
    conv2 = tf.layers.conv2d(inputs=conv1, filters=128, kernel_size=[2,2], strides=2, 
                             padding='same', activation = tf.nn.relu\
                             ,kernel_regularizer=regularizer)
    max_pool_2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=1, padding='same')

    
    conv3 = tf.layers.conv2d(inputs=max_pool_2, filters=256, kernel_size=[2,2], strides=2, 
                             padding='same', activation = tf.nn.relu\
                             ,kernel_regularizer=regularizer)
    max_pool_3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2,2], strides=1, padding='same')

    
    conv4 = tf.layers.conv2d(inputs=max_pool_3, filters=512, kernel_size=[2,2], strides=2, 
                             padding='same', activation = tf.nn.relu\
                             ,kernel_regularizer=regularizer)
    max_pool_4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2,2], strides=1, padding='same')
    
    
    


with graph.as_default():
    
    ####
    pool4_flat = tf.reshape(max_pool_4, [-1, 32768])
    
    dense = tf.layers.dense(inputs=pool4_flat, units=1024, activation=tf.nn.relu)
    
    dropout = tf.layers.dropout(inputs=dense, rate=(1-keep_prob_))
    
    logits = tf.layers.dense(inputs=dropout, units=n_classes)
    
#    ####
#    flat = tf.reshape(max_pool_4, (-1, 32768))
#    print (flat.shape)
#    flat = tf.nn.dropout(flat, keep_prob=keep_prob_)
#    
#    # Predictions
#    logits = tf.layers.dense(flat, n_classes)

    pred=tf.argmax(logits, 1)
    
    l2_loss = tf.losses.get_regularization_loss()
    # Cost function and optimizer
    cost = l2_loss+tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_))
    optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)
    
    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
    
if (os.path.exists('checkpoints-cnn2') == False):
    os.mkdir ('checkpoints-cnn2')
    
    


validation_acc = []
validation_loss = []

train_acc = []
train_loss = []

with graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1
   
    # Loop over epochs
    for e in range(epochs):
        
        # Loop over batches
        for x,y in get_batches(X_tr, y_tr, batch_size):
            
            print(e)
            y=y.toarray()
            # Feed dictionary
            feed = {inputs_ : x, labels_ : y, keep_prob_ : 0.6, learning_rate_ : learning_rate}
            
            
            
            loss = sess.run(cost, feed_dict = feed)
            
            
            grbz = sess.run( optimizer, feed_dict = feed)
            acc = sess.run( accuracy, feed_dict = feed)
            train_acc.append(acc)
            train_loss.append(loss)
            
            # Print at each 5 iters
            if (iteration % 5 == 0):
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {:d}".format(iteration),
                      "Train loss: {:6f}".format(loss),
                      "Train acc: {:.6f}".format(acc))
            
            # Compute validation loss at every 10 iterations
            if (iteration%10 == 0):                
                val_acc_ = []
                val_loss_ = []
                
                for x_v, y_v in get_batches(X_vld, y_vld, batch_size_V):
                    # Feed
                    y_v=y_v.toarray()
                    feed = {inputs_ : x_v, labels_ : y_v, keep_prob_ : 1.0}  
                    
                    # Loss
                    loss_v= sess.run(cost, feed_dict = feed)
                    acc_v = sess.run(accuracy, feed_dict = feed)                    
                    val_acc_.append(acc_v)
                    val_loss_.append(loss_v)
                
                # Print info
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {:d}".format(iteration),
                      "Validation loss: {:6f}".format(np.mean(val_loss_)),
                      "Validation acc: {:.6f}".format(np.mean(val_acc_)))
                
                # Store
                validation_acc.append(np.mean(val_acc_))
                validation_loss.append(np.mean(val_loss_))
            
            # Iterate 
            iteration += 1
        saver.save(sess,"checkpoints-cnn2/har.ckpt")
    





#########


with graph.as_default():
    saver = tf.train.Saver()
    

def get_max_oc(lst):
    
    olo=max(set(lst), key=lst.count)
    return olo


labels_ALL = []
validation_label = []
w=0
batch_size_ALL=1;

with tf.Session(graph=graph) as sess:
    # Restore
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints-cnn2'))
    
    for name in validation_file_list:
        
        
        XM,SR=librosa.core.load(name)
        XM=XM[int(0.15*len(XM)):int(0.85*len(XM))]
        XM_frame=librosa.util.frame(XM, frame_length=int(SR*3)\
                                  , hop_length=int(SR*3))
    
    
        X_test2=[]
        for i in range(XM_frame.shape[1]):
            
            print(w)
            kk=XM_frame[:,i]
            
            mfcc_cc = librosa.core.amplitude_to_db(librosa.feature.melspectrogram(y=kk, sr=SR,n_mels=128,\
                                                     fmax=8000,hop_length=int(SR*20/1000)),ref=1)
            w=w+1
            
            mfcc_bb=mfcc_cc[:,:128]
            
            cur_image=mfcc_bb
            mval=np.mean(cur_image)
            sval=np.std(cur_image)
            new_image=(cur_image-mval)/(sval+0.00000001)
            
            minval=np.min(new_image)
            maxval=np.max(new_image)
            
            new_image2=(new_image-minval)/(maxval-minval)
            X_test2.append(new_image2)
        
        X_test2=np.array(X_test2)
        
        X_test3=X_test2.reshape(-1,128,128,1)
        all_pred=[]
        for x_t in get_batches_X_ALL(X_test3,batch_size_ALL):
            
            feed = {inputs_: x_t,
                    keep_prob_: 1}
            
            pred_val = sess.run(pred, feed_dict=feed)
            all_pred.append(pred_val)

        
        
        op=[int(x) for x in all_pred]
        max_oc=get_max_oc(op)
        
        print(max_oc)
        print('\n')
        validation_label.append(max_oc)
    
        


output_file='CNN_4_result.csv'

# appending the test data sample id and predicted data sample class label in a list
table=[]
ipd=str('id')
cc=str('class')
table.append([ipd,cc]) 
for i,name in enumerate(validation_file_list):
    y=inverse_CC[validation_label[i]]
    j=os.path.basename(name)
    table.append([j,y])


# creating a csv file of data sample id and their corresponding  class label 
with open(output_file, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in table:
        writer.writerow(val)