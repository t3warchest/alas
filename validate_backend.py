#!/usr/bin/env python3
"""
validate_backend.py — ALAS backend integration layer standalone validator
100 tests. Zero external dependencies. Run: python3 validate_backend.py
"""
import asyncio, json, logging, os, sys, tempfile, time, threading
sys.path.insert(0, os.path.dirname(__file__))
os.environ.update({'JWT_SECRET':'test-secret-validate-backend','DEMO_MODE':'true','JWT_EXPIRY_SECONDS':'3600','REQUIRE_AUTH':'false','PERSIST_SESSIONS':'false','GATEWAY_PORT':'8001','AGENT_SERVICE_URL':'http://localhost:8000'})
logging.disable(logging.CRITICAL)

G='\033[92m'; R='\033[91m'; C='\033[96m'; B='\033[1m'; DIM='\033[2m'; X='\033[0m'
_c=[0,0]
def ok(n):   _c[0]+=1; print(f'  {G}✓{X} {n}')
def fail(n,r=''): _c[1]+=1; print(f'  {R}✗{X} {n}'); r and print(f'    {DIM}{r}{X}')
def section(t): print(f'\n{B}{C}▶ {t}{X}')
def ae(n,g,e):  (ok(n) if g==e else fail(n,f'got {g!r} expected {e!r}'))
def at(n,c,h=''): (ok(n) if c else fail(n,h or 'False'))
def ar(n,fn,exc=Exception):
    try: fn(); fail(n,'no exc raised')
    except exc: ok(n)
    except Exception as e: fail(n,f'wrong exc {type(e).__name__}: {e}')
def run(n,fn):
    try: fn()
    except Exception as e:
        import traceback; fail(n,f'{type(e).__name__}: {e}\n    {traceback.format_exc().splitlines()[-2]}')
def arun(coro):
    try: return asyncio.run(coro)
    except RuntimeError:
        loop=asyncio.new_event_loop()
        try: return loop.run_until_complete(coro)
        finally: loop.close()

section('Shared Utils — Logging')
from shared.utils.logging import get_logger
def t(): at('get_logger returns logger', get_logger('x') is not None)
def t2(): at('same name same instance', get_logger('y') is get_logger('y'))
def t3(): at('has handler', len(get_logger('z').handlers) > 0)
run('get_logger',t); run('same instance',t2); run('has handler',t3)

section('Event Constants')
from shared.contracts.events import CLIENT_EVENTS, SERVER_EVENTS, WS_EVT_TOKEN, WS_EVT_TURN_DONE, WS_EVT_SCORE, WS_EVT_PHASE_CHANGE, WS_EVT_SESSION_END, WS_EVT_PONG, WS_EVT_ERROR, WS_EVT_FATAL, WS_EVT_AUTH_OK, WS_EVT_CONNECTED, WS_MSG_MESSAGE, WS_MSG_PING, WS_MSG_END, WS_MSG_RECONNECT
at('no overlap', not (CLIENT_EVENTS & SERVER_EVENTS))
ae('token',WS_EVT_TOKEN,'token'); ae('turn_done',WS_EVT_TURN_DONE,'turn_done'); ae('score',WS_EVT_SCORE,'score')
ae('phase_change',WS_EVT_PHASE_CHANGE,'phase_change'); ae('session_end',WS_EVT_SESSION_END,'session_end')
ae('error',WS_EVT_ERROR,'error'); ae('fatal',WS_EVT_FATAL,'fatal'); ae('pong',WS_EVT_PONG,'pong')
ae('auth_ok',WS_EVT_AUTH_OK,'auth_ok'); ae('connected',WS_EVT_CONNECTED,'connected')
ae('msg_message',WS_MSG_MESSAGE,'message'); ae('msg_ping',WS_MSG_PING,'ping'); ae('msg_end',WS_MSG_END,'end'); ae('msg_reconnect',WS_MSG_RECONNECT,'reconnect')

section('Message Serialisation')
from shared.contracts.messages import (WSMessage, ClientMessage, ClientPing, ClientReconnect, TokenEvent, TurnDoneEvent, ScoreEvent, PhaseChangeEvent, SessionEndEvent, PongEvent, ErrorEvent, FatalEvent, AuthOkEvent, ConnectedEvent, parse_client_frame)
def t(): d=json.loads(WSMessage(type='test').to_json()); ae('ws_msg_json',d['type'],'test')
def t2(): d=TokenEvent(content='hi',turn_index=3).to_dict(); ae('token_content',d['content'],'hi'); ae('token_turn',d['turn_index'],3)
def t3(): d=TurnDoneEvent(session_id='s1',emotion='curious',phase='core').to_dict(); ae('turn_done_emotion',d['emotion'],'curious')
def t4(): d=ScoreEvent(composite=0.76,rationale='Good.').to_dict(); at('score_composite',abs(d['composite']-0.76)<0.001)
def t5(): d=PhaseChangeEvent(from_phase='setup',to_phase='core').to_dict(); ae('phase_change_from',d['from_phase'],'setup')
def t6(): d=SessionEndEvent(session_id='s1',summary={'trend':'improving'}).to_dict(); ae('session_end_summary',d['summary']['trend'],'improving')
def t7(): d=ErrorEvent(code='e',message='m',recoverable=True).to_dict(); at('error_recoverable',d['recoverable'] is True)
def t8(): d=FatalEvent(code='f',message='m').to_dict(); at('fatal_not_recoverable',d['recoverable'] is False)
def t9(): d=AuthOkEvent(user_id='u1',token_expires_in=3600).to_dict(); ae('auth_ok_user_id',d['user_id'],'u1')
def t10(): at('pong_ts',PongEvent(timestamp=1234.5).to_dict()['timestamp']==1234.5)
run('ws_message json',t); run('token event',t2); run('turn_done event',t3); run('score event',t4)
run('phase_change event',t5); run('session_end event',t6); run('error recoverable',t7)
run('fatal not recoverable',t8); run('auth_ok',t9); run('pong timestamp',t10)

section('parse_client_frame')
def t(): f=parse_client_frame(json.dumps({'type':'message','content':'Hello!'})); at('is_ClientMessage',isinstance(f,ClientMessage)); ae('content',f.content,'Hello!')
def t2(): f=parse_client_frame(json.dumps({'type':'ping','timestamp':9.0})); at('is_ClientPing',isinstance(f,ClientPing)); ae('ts',f.timestamp,9.0)
def t3(): f=parse_client_frame(json.dumps({'type':'reconnect','session_id':'abc','last_turn_index':5})); at('is_ClientReconnect',isinstance(f,ClientReconnect)); ae('sid',f.session_id,'abc')
def t4(): at('invalid_json→None',parse_client_frame('not json') is None)
def t5(): at('empty→None',parse_client_frame('') is None)
def t6(): f=parse_client_frame(json.dumps({'type':'custom'})); at('unknown_type',f is not None and f.type=='custom')
run('parse message',t); run('parse ping',t2); run('parse reconnect',t3); run('invalid json→None',t4); run('empty→None',t5); run('unknown type',t6)

section('GatewaySession Model')
from shared.contracts.session import GatewaySession, SessionStatus
def gs(**kw): return GatewaySession(session_id='s1',user_id='u1',scenario_id='job',**kw)
def t(): ae('initial_status',gs().status,SessionStatus.INITIALISING)
def t2(): s=gs(); s.mark_active(); s.mark_disconnected(); s.mark_ended(); ae('ended',s.status,SessionStatus.ENDED)
def t3(): s=gs(); s.mark_disconnected(); at('disconnected_reconnectable',s.is_reconnectable())
def t4(): s=gs(); s.mark_ended(); at('ended_not_reconnectable',not s.is_reconnectable())
def t5(): s=gs(); s.mark_disconnected(); s.last_active_at=time.time()-7201; at('expired_not_reconnectable',not s.is_reconnectable())
def t6(): s=gs(scenario_title='T',persona_name='A',current_phase='core',last_turn_index=4,total_turns=5,latest_composite=0.71); s.mark_active(); r=GatewaySession.from_dict(s.to_dict()); ae('persona_name',r.persona_name,'A'); ae('phase',r.current_phase,'core'); at('composite',abs((r.latest_composite or 0)-0.71)<0.001)
def t7(): s=gs(); s.mark_active(); at('status_is_string',isinstance(s.to_dict()['status'],str))
run('initial status',t); run('lifecycle',t2); run('disconnected reconnectable',t3); run('ended not reconnectable',t4); run('expired not reconnectable',t5); run('roundtrip',t6); run('status string',t7)

section('InMemorySessionStore')
from gateway.session_store.store import InMemorySessionStore
def mg(sid='s1',uid='u1'): return GatewaySession(session_id=sid,user_id=uid,scenario_id='j')
def t(): s=InMemorySessionStore(); s.create(mg('a')); r=s.get('a'); at('create_get',r is not None); ae('session_id',r.session_id,'a')
def t2(): s=InMemorySessionStore(); at('get_missing',s.get('x') is None)
def t3(): s=InMemorySessionStore(); s.create(mg('u')); g=s.get('u'); g.current_phase='core'; s.update(g); ae('update',s.get('u').current_phase,'core')
def t4(): s=InMemorySessionStore(); s.create(mg('d')); s.delete('d'); at('delete',s.get('d') is None)
def t5(): s=InMemorySessionStore(); s.create(mg('a','alice')); s.create(mg('b','alice')); s.create(mg('c','bob')); at('list_user',len(s.list_for_user('alice'))==2)
def t6(): s=InMemorySessionStore(); s.create(mg('t')); s.update_turn('t',turn_index=5,phase='esc',composite=0.68); r=s.get('t'); ae('turn_idx',r.last_turn_index,5); ae('phase',r.current_phase,'esc')
def t7():
    s=InMemorySessionStore(); old=mg('old'); old.mark_disconnected(); old.last_active_at=time.time()-9999; s.create(old)
    fresh=mg('fresh'); fresh.mark_disconnected(); s.create(fresh)
    removed=s.expire_old_sessions(ttl_seconds=3600); at('expire_removed',removed==1); at('old_gone',s.get('old') is None); at('fresh_kept',s.get('fresh') is not None)
def t8():
    s=InMemorySessionStore(); errors=[]
    def cm(prefix):
        for i in range(50):
            try: s.create(mg(f'{prefix}-{i}',uid=prefix))
            except Exception as e: errors.append(str(e))
    threads=[threading.Thread(target=cm,args=(f't{i}',)) for i in range(4)]
    for th in threads: th.start()
    for th in threads: th.join()
    at('thread_no_errors',not errors); at('thread_count',len(s)==200)
run('create_get',t); run('get_missing',t2); run('update',t3); run('delete',t4); run('list_user',t5); run('update_turn',t6); run('expire',t7); run('thread_safety',t8)

section('SQLiteSessionStore')
from gateway.session_store.store import SQLiteSessionStore
def t():
    with tempfile.TemporaryDirectory() as d:
        s=SQLiteSessionStore(db_path=f'{d}/t.db'); s.create(mg('sq1')); r=s.get('sq1'); at('sq_create_get',r is not None); ae('sq_sid',r.session_id,'sq1')
def t2():
    with tempfile.TemporaryDirectory() as d:
        db=f'{d}/p.db'; s1=SQLiteSessionStore(db_path=db); s1.create(mg('p')); s1.update_status('p',SessionStatus.ACTIVE)
        s2=SQLiteSessionStore(db_path=db); r=s2.get('p'); at('sq_reopen',r is not None and r.status==SessionStatus.ACTIVE)
def t3():
    with tempfile.TemporaryDirectory() as d:
        s=SQLiteSessionStore(db_path=f'{d}/u.db'); s.create(mg('u2')); g=s.get('u2'); g.current_phase='core'; s.update(g); ae('sq_update',s.get('u2').current_phase,'core')
def t4():
    with tempfile.TemporaryDirectory() as d:
        s=SQLiteSessionStore(db_path=f'{d}/d.db'); s.create(mg('del2')); s.delete('del2'); at('sq_delete',s.get('del2') is None)
def t5():
    with tempfile.TemporaryDirectory() as d:
        s=SQLiteSessionStore(db_path=f'{d}/l.db')
        for i in range(5): s.create(mg(f'b-{i}','bulk'))
        at('sq_list',len(s.list_for_user('bulk'))==5)
run('sq create_get',t); run('sq survives reopen',t2); run('sq update',t3); run('sq delete',t4); run('sq list_user',t5)

section('JWT Handler')
from gateway.auth.jwt_handler import create_token, verify_token, extract_user_id, issue_demo_token, _b64url_encode, _b64url_decode
def t(): data=b'hello\x00\xff'; ae('b64_rt',_b64url_decode(_b64url_encode(data)),data)
def t2(): at('no_padding','=' not in _b64url_encode(b'a'))
def t3(): at('three_parts',len(create_token('u').split('.'))==3)
def t4(): ae('sub_claim',verify_token(create_token('alice'))['sub'],'alice')
def t5(): at('default_role','student' in verify_token(create_token('u'))['roles'])
def t6(): at('exp_future',verify_token(create_token('u',expires_in=300))['exp']>int(time.time()))
def t7(): ar('expired_raises',lambda: verify_token(create_token('u',expires_in=-1)),ValueError)
def t8(): h,p,_=create_token('u').split('.'); ar('tampered_sig',lambda: verify_token(f'{h}.{p}.bad'),ValueError)
def t9(): ar('malformed',lambda: verify_token('not.a.token'),ValueError)
def t10(): at('jti_unique',verify_token(create_token('u'))['jti']!=verify_token(create_token('u'))['jti'])
def t11(): ae('extract_uid',extract_user_id(create_token('bob')),'bob')
def t12(): r=issue_demo_token('du'); ae('demo_uid',r['user_id'],'du'); at('demo_valid',verify_token(r['access_token'])['sub']=='du')
run('b64 roundtrip',t); run('no padding',t2); run('3 parts',t3); run('sub claim',t4); run('default role',t5); run('exp future',t6); run('expired raises',t7); run('tampered sig',t8); run('malformed',t9); run('jti unique',t10); run('extract_user_id',t11); run('demo token',t12)

section('AgentServiceClient — stream_turn')
from gateway.services.agent_client import AgentServiceClient, AgentServiceError
def _r(**kw):
    b={'session_id':'s1','turn_index':1,'avatar_response':'Alpha Beta Gamma','emotion':'curious','scenario_phase':'core','session_ended':False,'turn_score':{'turn_index':1,'clarity':0.8,'empathy':0.7,'structure':0.75,'relevance':0.85,'confidence':0.7,'composite':0.76,'rationale':'Good.'},'session_summary':None}
    b.update(kw); return b
async def _col(gen): return [e async for e in gen]
def _patch(c,result=None,error=None):
    async def _f(*a,**kw):
        if error: raise error
        return result
    c.send_message=_f
def t():
    c=AgentServiceClient('http://x'); _patch(c,_r())
    events=arun(_col(c.stream_turn('s1','hi'))); tokens=[e for e in events if e['type']==WS_EVT_TOKEN]; at('yields_tokens',len(tokens)==3)
def t2():
    c=AgentServiceClient('http://x'); _patch(c,_r())
    events=arun(_col(c.stream_turn('s1','hi'))); done=[e for e in events if e['type']==WS_EVT_TURN_DONE]; at('1_turn_done',len(done)==1); ae('emotion',done[0]['emotion'],'curious')
def t3():
    c=AgentServiceClient('http://x'); _patch(c,_r())
    events=arun(_col(c.stream_turn('s1','hi'))); scores=[e for e in events if e['type']==WS_EVT_SCORE]; at('1_score',len(scores)==1); at('composite',abs(scores[0]['composite']-0.76)<0.001)
def t4():
    c=AgentServiceClient('http://x'); _patch(c,_r(turn_score=None))
    events=arun(_col(c.stream_turn('s1','hi'))); at('no_score_none',not any(e['type']==WS_EVT_SCORE for e in events))
def t5():
    c=AgentServiceClient('http://x'); _patch(c,_r(session_ended=True,session_summary={'trend':'improving'}))
    events=arun(_col(c.stream_turn('s1','hi'))); ends=[e for e in events if e['type']==WS_EVT_SESSION_END]; at('session_end',len(ends)==1); ae('trend',ends[0]['summary']['trend'],'improving')
def t6():
    c=AgentServiceClient('http://x'); _patch(c,error=AgentServiceError('down',503))
    events=arun(_col(c.stream_turn('s1','hi'))); at('error_event',len(events)==1 and events[0]['type']=='error')
def t7():
    c=AgentServiceClient('http://x'); _patch(c,_r(avatar_response='A B C'))
    events=arun(_col(c.stream_turn('s1','hi'))); types=[e['type'] for e in events]
    lt=max(i for i,t in enumerate(types) if t==WS_EVT_TOKEN); fd=next(i for i,t in enumerate(types) if t==WS_EVT_TURN_DONE); at('tokens_before_done',lt<fd)
run('stream tokens',t); run('stream turn_done',t2); run('stream score',t3); run('no score None',t4); run('session_end',t5); run('agent error',t6); run('tokens before done',t7)

section('Gateway Config')
from gateway.config import get_gateway_settings
def t(): s=get_gateway_settings(); ae('port',s.port,8001); at('agent_url','localhost:8000' in s.agent_base_url); at('demo',s.demo_mode is True)
def t2(): at('cached',get_gateway_settings() is get_gateway_settings())
run('config loads',t); run('config cached',t2)

section('WebSocket Auth Helper')
from gateway.routers.websocket import _authenticate
class _FWS:
    def __init__(self): self.sent=[]
    async def send_text(self,d): self.sent.append(json.loads(d))
    async def close(self,code=None): pass
def t(): ws=_FWS(); result=arun(_authenticate(ws,'','sess-xyz')); at('demo_no_token',result is not None); at('demo_has_id',bool(result))
def t2(): token=create_token('test-ws'); ws=_FWS(); result=arun(_authenticate(ws,token,'any')); ae('valid_token',result,'test-ws')
def t3(): token=create_token('u',expires_in=-1); ws=_FWS(); result=arun(_authenticate(ws,token,'any')); at('expired→None',result is None); at('expired_fatal',ws.sent and ws.sent[0]['type']=='fatal')
def t4(): ws=_FWS(); result=arun(_authenticate(ws,'garbage.bad.token','any')); at('bad→None',result is None); at('bad_fatal',ws.sent and ws.sent[0]['type']=='fatal')
run('demo no token',t); run('valid token',t2); run('expired→None+fatal',t3); run('bad→None+fatal',t4)

p,f=_c; total=p+f
print(f'\n{"━"*62}')
print(f'{B}Results:{X}  {G}{p} passed{X}  {(R if f else DIM)}{f} failed{X}  of {total} total')
print(f'{"━"*62}\n')
sys.exit(1 if f else 0)
