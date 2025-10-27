import sqlite3,json
conn=sqlite3.connect('users.sqlite3')
cur=conn.cursor()
cur.execute('SELECT id, analysis_json FROM analyses ORDER BY datetime(created_at) DESC LIMIT 1')
r=cur.fetchone()
if not r:
    print('no rows')
else:
    print('id=',r[0])
    aj=r[1]
    print('len',len(aj))
    obj=json.loads(aj)
    print('keys:', list(obj.keys()))
    for k in ['what_has_happened','key_legal_points','general_follow_ups','disclaimer']:
        v=obj.get(k)
        t=type(v).__name__
        print('\nkey:',k,'type=',t)
        if isinstance(v,(list,tuple)):
            print('count=',len(v),'sample=',v[:3])
        else:
            s=str(v or '')
            print('len=',len(s),'sample=',s[:300])
conn.close()
