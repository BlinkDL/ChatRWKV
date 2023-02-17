
import copy
from .model_run import RWKV_RNN
from .utils import TOKENIZER

class RWKVContext:
    def __init__(self) -> None:
        self.tokens = []
        self.state = None
        self.last_out = None
        
    def __str__(self) -> str:
        return f"""
    Tokens:   {str(self.tokens)}
    State:    Tensor of shape {str(self.state.shape)}
    Last Out: Tensor of shape {str(self.last_out.shape)}
              """

class ChatRWKV:
    def __init__(self, args, tokenizer_name="ChatRWKV/20B_tokenizer.json") -> None:
        AVOID_REPEAT = '，。：？！'
        print(f'Loading model - {args.MODEL_NAME}')
        self.model = RWKV_RNN(args)
        self.tokenizer = TOKENIZER(tokenizer_name, args.RUN_DEVICE)
        self.avoid_repeat_tokens = []
        for i in AVOID_REPEAT:
            dd = self.tokenizer.encode(i)
            assert len(dd) == 1
            self.avoid_repeat_tokens += dd
        self.change_temp(args.GEN_TEMP)
        self.change_top_p(args.GEN_TOP_P)
        self.ctx_len = args.ctx_len
    
    def change_temp(self, x_temp):
        if x_temp <= 0.2:
            x_temp = 0.2
        if x_temp >= 5:
            x_temp = 5
        self.x_temp = x_temp
        
    def change_top_p(self, x_top_p):
        if x_top_p <= 0:
            x_top_p = 0 
        self.x_top_p = x_top_p
        
    def _run_rnn_impl(self, tokens, ctx: RWKVContext = RWKVContext(), newline_adj=0):
        tokens = [int(x) for x in tokens]
        ctx.tokens += tokens
        out, ctx.state = self.model.forward(tokens, ctx.state)

        out[0] = -999999999  # disable <|endoftext|>
        out[187] += newline_adj  # adjust \n probability
        if self.avoid_repeat_tokens and ctx.tokens[-1] in self.avoid_repeat_tokens:
            out[ctx.tokens[-1]] = -999999999
        ctx.last_out = out
        return out, ctx

    def generate(self, ctx: RWKVContext = RWKVContext(), newline_adj=0):
        tmp_token = []
        while True: 
            token = self.tokenizer.sample_logits(
                ctx.last_out,
                ctx.tokens,
                self.ctx_len,
                temperature=self.x_temp,
                top_p=self.x_top_p,
            )
            _, ctx = self._run_rnn_impl([token], ctx, newline_adj)
            tmp_token.append(token)
            xxx = self.tokenizer.decode(tmp_token)
            if '\ufffd' not in xxx: 
                break
        return xxx, copy.deepcopy(ctx)

    def get_context(self, input, ctx: RWKVContext = RWKVContext(), newline_adj=0):
        _, ctx = self._run_rnn_impl(
            self.tokenizer.encode(input), newline_adj=newline_adj)
        return copy.deepcopy(ctx)