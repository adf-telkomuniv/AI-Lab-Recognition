from keras import backend as K
import cv2
K.set_image_data_format('channels_first')
from utils.inception_blocks_v2 import *
np.set_printoptions(threshold=np.nan)


def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = alpha + pos_dist - neg_dist
    loss = tf.reduce_sum(tf.maximum(0., basic_loss))
    return loss

print('pre')
FRmodel = faceRecoModel(input_shape=(3, 96, 96))
FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
FRmodel.load_weights('weights.h5')


database = {}
print('start')

def img_to_encoding(image_path, model):
    img1 = cv2.imread(image_path, 1)
    img = img1[..., ::-1]
    img = np.around(np.transpose(img, (2, 0, 1)) / 255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding


with tf.Session() as test:
    tf.set_random_seed(1)
    y_true = (None, None, None)
    y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed=1),
              tf.random_normal([3, 128], mean=1, stddev=1, seed=1),
              tf.random_normal([3, 128], mean=3, stddev=4, seed=1))
    loss = triplet_loss(y_true, y_pred)

    print("loss = " + str(loss.eval()))

print('start 2')
# database["danielle"] = img_to_encoding("faces/danielle.png", FRmodel)
# database["younes"] = img_to_encoding("faces/younes.jpg", FRmodel)
# database["tian"] = img_to_encoding("faces/tian.jpg", FRmodel)
# database["andrew"] = img_to_encoding("faces/andrew.jpg", FRmodel)
# database["kian"] = img_to_encoding("faces/kian.jpg", FRmodel)
# database["dan"] = img_to_encoding("faces/dan.jpg", FRmodel)
# database["sebastiano"] = img_to_encoding("faces/sebastiano.jpg", FRmodel)
# database["bertrand"] = img_to_encoding("faces/bertrand.jpg", FRmodel)
# database["kevin"] = img_to_encoding("faces/kevin.jpg", FRmodel)
# database["felix"] = img_to_encoding("faces/felix.jpg", FRmodel)
# database["benoit"] = img_to_encoding("faces/benoit.jpg", FRmodel)
# database["arnaud"] = img_to_encoding("faces/arnaud.jpg", FRmodel)
database["halprin"] = img_to_encoding("faces/halprin.jpg", FRmodel)
database["bintang"] = img_to_encoding("faces/bintang.jpg", FRmodel)
database["roki"] = img_to_encoding("faces/roki.jpg", FRmodel)

def verify(image_path, identity, database, model):
    """
    Function that verifies if the person on the "image_path" image is "identity".

    Arguments:
    image_path -- path to an image
    identity -- string, name of the person you'd like to verify the identity. Has to be a resident of the Happy house.
    database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
    model -- your Inception model instance in Keras

    Returns:
    dist -- distance between the image_path and the image of "identity" in the database.
    door_open -- True, if the door should open. False otherwise.
    """

    ### START CODE HERE ###

    # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
    encoding = img_to_encoding(image_path, model)

    # Step 2: Compute distance with identity's image (≈ 1 line)
    dist = np.linalg.norm(encoding - database[identity])

    # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
    if dist<0.7:
        print("It's " + str(identity) + ", welcome home!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False

    ### END CODE HERE ###

    return dist, door_open

verify("faces/test02.jpg", "halprin", database, FRmodel)
verify("faces/test01.jpg", "halprin", database, FRmodel)
verify("faces/test03.jpg", "roki", database, FRmodel)


def who_is_it(image_path, database, model):
    """
    Implements face recognition for the happy house by finding who is the person on the image_path image.

    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras

    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """

    ### START CODE HERE ###

    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
    encoding = img_to_encoding(image_path, model)

    ## Step 2: Find the closest encoding ##

    # Initialize "min_dist" to a large value, say 100 (≈1 line)
    min_dist = 100

    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():

        # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
        dist = np.linalg.norm(encoding - db_enc)

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if dist < min_dist:
            min_dist = dist
            identity = name

    ### END CODE HERE ###

    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print("it's " + str(identity) + ", the distance is " + str(min_dist))

    return min_dist, identity

who_is_it("faces/test03.jpg", database, FRmodel)