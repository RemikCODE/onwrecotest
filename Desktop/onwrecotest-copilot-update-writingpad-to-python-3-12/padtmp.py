import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from hwr.models.ONHWRECO import ONHWRECO
from hwr.decoding.ctc_decoder import BestPathDecoder
from hwr.data.generator import pad_2d

m = ONHWRECO(preload=True, gru=False, decoder=BestPathDecoder())
d = np.load(r'npz-6\e04\e04-196\e04-196z-06.npz')
x = d['x'].astype('float32')

for pad in [0, 10, 20]:
    if pad == 0:
        xi = x
    else:
        xi = pad_2d(x, pad_to=x.shape[0] + pad, pad_value=0)
    length = int(np.ceil(x.shape[0] / 4))
    sm = m.pred_model.predict(np.expand_dims(xi, 0), verbose=0)[0]
    pred = BestPathDecoder().decode([sm], top_n=1, input_lengths=[min(length, sm.shape[0])])[0][0]
    print("pad=%d  -> %r" % (pad, pred))

print("AUTH : and pinned do khe fame")