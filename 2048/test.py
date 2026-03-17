import model as G

G.reset()
print(233)
num = 0
for i in range(10000):
    # tmp=int(input())
    tmp = i
    _, _, done = G.RL_step(tmp % 4)
    if done:
        G.reset()
        num += 1
    # print(flag)
print(num)
