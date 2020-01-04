def get_params(deep=True, cnt = 0):
  if not deep:
    cnt = cnt + 1
    print("test")
    if(cnt > 3):
      return 
    return get_params(deep=False, cnt = cnt)

get_params(deep = False)