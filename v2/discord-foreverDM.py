from chatForeverDM import user, bot, interface, on_message, out, srv, FREE_GEN_LEN, pipeline, load_all_stat, save_all_stat, GEN_TEMP, GEN_TOP_P, END_OF_TEXT, run_rnn, GEN_alpha_presence, GEN_alpha_frequency
import pandas as pd
import random
import nest_asyncio
import interactions
import os
from dotenv import load_dotenv
import os
from rwkv.model import RWKV
from rwkv.utils import PIPELINE
load_dotenv()
nest_asyncio.apply()


ins = '''\n\n###Instruction You are the DM for a game that's a cross between Dungeons and Dragons and a choose-your-own-adventure game. You will be given an action and a sentence about how that action goes. You will send me an immersive and detailed response describing how the action went for [char].
### Input:'''
# srv = 'dummy_server'
delim = '##########'
TOKEN = os.getenv('DISCORD_TOKEN')

die=pd.read_csv('/root/foreverDM/newRollingTable.csv',index_col=None)



client = interactions.Client()

@interactions.listen()
async def on_ready():
    print(f"Logged in as {client.user} (ID: {client.user.id}), {client.status}")
    print("------")
    
@interactions.slash_command("dm", description="summon foreverDM!")
@interactions.slash_option(
    "msg",
    "send foreverDM a sentence with a clear action, and she'll tell you the outcome!",
    opt_type=interactions.OptionType.STRING,
    required=True)
async def dm(ctx: interactions.SlashContext,msg:str):
    OGmsg = msg 
    global tokenString
    await ctx.defer()
    srv = str(ctx.author.username)   
    try:
      out = load_all_stat(srv, 'chat')
    except: #new user
      out = load_all_stat('', 'chat_init')
      save_all_stat(srv, 'chat', out)
    # output = on_message(msg,srv,FREE_GEN_LEN = 100)
    if '+' in msg:
      print('command')
      output = on_message(msg,srv,FREE_GEN_LEN = 100)
      if output is None:
        output = 'Chat reset!'
      await ctx.send(f"**Input**: {OGmsg}\n**Response**: {output.replace('#','')}")
    else:
      print('game')
      r=random.randint(0,19)
      roll=die['sentence'][r]
      
      result = f'\nresult: {roll}'
      msg = f'{msg}{result}'
      output = on_message(msg,srv,FREE_GEN_LEN = 100)
      await ctx.send(f"Action: {OGmsg}\n**Dice: {die['die'][r]}**\n{output.replace('#','')}")
# await ctx.send(f"You input {msg}")

client.start(TOKEN)