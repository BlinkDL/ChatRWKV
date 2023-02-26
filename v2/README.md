TRIAL 2

```
Check LAMBADA... RWKV_PRELOADING 0
100 ppl 5.99 acc 60.0
200 ppl 4.97 acc 65.5
300 ppl 5.13 acc 64.67
400 ppl 5.57 acc 62.25
500 ppl 5.49 acc 62.4
>>> Timespent: 311.30 seconds

Check LAMBADA... RWKV_PRELOADING 1 (new)
100 ppl 5.99 acc 60.0
200 ppl 4.97 acc 65.5
300 ppl 5.13 acc 64.67
400 ppl 5.57 acc 62.25
500 ppl 5.49 acc 62.4
>>> Timespent: 308.63 seconds

Check LAMBADA... RWKV_PRELOADING 1 (old)
100 ppl 5.99 acc 60.0
200 ppl 4.97 acc 65.5
300 ppl 5.13 acc 64.67
400 ppl 5.57 acc 62.25
500 ppl 5.49 acc 62.4
>>> Timespent: 305.26 seconds
```

- - -

TRIAL 1

# 3b
Before preloading
```
Check LAMBADA...
100 ppl 5.99 acc 60.0
200 ppl 4.97 acc 65.5
300 ppl 5.13 acc 64.67
400 ppl 5.57 acc 62.25
500 ppl 5.49 acc 62.4
>>> Timespent: 334.68 seconds
```

After preloading
```
Check LAMBADA...
100 ppl 5.99 acc 60.0
200 ppl 4.97 acc 65.5
300 ppl 5.13 acc 64.67
400 ppl 5.57 acc 62.25
500 ppl 5.49 acc 62.4
>>> Timespent: 316.82 seconds
```

# 7b

Before preloading
```
Check LAMBADA...
100 ppl 4.74 acc 70.0
>>> Timespent: 142.19 seconds
```

After preloading
```
Check LAMBADA...
100 ppl 4.74 acc 70.0
>>> Timespent: 140.82 seconds
```

# 1b5
Before preloading
```
Check LAMBADA...
100 ppl 9.04 acc 55.0
200 ppl 7.01 acc 62.5
300 ppl 6.88 acc 62.33
>>> Timespent: 88.39 seconds
```

After preloading
```
Check LAMBADA...
100 ppl 9.04 acc 55.0
200 ppl 7.01 acc 62.5
300 ppl 6.88 acc 62.33
>>> Timespent: 87.75 seconds
```
