import jax
import jax.numpy as jnp
import math
import numpy as np
from lte_code.modified_flax_gtp2 import modFlaxGPT2Module
import flax.linen as nn

class LTE:
    def __init__(self, config, seed=0):
        self.config = config
        self.embed_reward = nn.Dense(config.hidden_size)
        self.embed_action = nn.Dense(config.hidden_size)
#         self.embed_special = nn.Dense(config.hidden_size)
      
        self.pred_max = nn.Dense(config.act_dim)
        self.pred_exp = nn.Dense(config.act_dim)
        self.pred_nonmax = nn.Dense(config.act_dim)
        self.pred_nonexp = nn.Dense(config.act_dim)

        self.seq_module = modFlaxGPT2Module(config)
        self.layern = nn.LayerNorm(use_bias=False, use_scale=False)
        
    def init(self, key):
        key1, key2, key3, key4, key5, key6, key7, key8, key9 = jax.random.split(key, 9)
        
        # the embedding parameters
        er = self.embed_reward.init(key1, jnp.arange(1))
        ea = self.embed_action.init(key2, jnp.arange(self.config.act_dim))

        # the prediction parameters 
        pm = self.pred_max.init(key4, jnp.arange(self.config.hidden_size))
        pe = self.pred_exp.init(key5, jnp.arange(self.config.hidden_size))
        pnm = self.pred_nonmax.init(key6, jnp.arange(self.config.hidden_size))
        pne = self.pred_nonexp.init(key7, jnp.arange(self.config.hidden_size))
        seq = self.seq_module.init(key8,
            input_embeds= jnp.zeros((1, 1,
                                     self.config.hidden_size)),
            attention_mask = jnp.zeros((1,1), dtype=bool),
            position_ids = jnp.zeros((1,1), dtype=int))
        ln = self.layern.init(rngs=key9, x=jnp.ones(self.config.hidden_size))
        return {'embed_reward' : er, 'embed_action' : ea,
                'pred_max' : pm, 'pred_exp' : pe,
                'pred_nonmax' : pnm, 'pred_nonexp' : pne,
                'seq' : seq, 'ln' : ln}


    def apply(self, params, actions, rewards, position_ids,
              **kwargs):
        """embed the actions and rewards (and later on states) together as a single token and feed it in"""
        # embbed both and add them
        embeddings = (self.embed_reward.apply(params['embed_reward'], rewards)
                      + self.embed_action.apply(params['embed_action'], actions))
        embeddings = self.layern.apply(params['ln'], embeddings)
        ret = self.seq_module.apply(params['seq'],
                                    input_embeds=embeddings,
                                    position_ids=position_ids,
                                    attention_mask=None,
                                    **kwargs)
        return ret