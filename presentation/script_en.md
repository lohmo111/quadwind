# Presentation script — “The drone that remembers the wind” (team 8)

Two speakers: **A** — slides 1–4 and 9 (opens and closes), **B** — slides 5–8 (the algorithm).

## 1. The drone that remembers the wind — Speaker A (~22 s)

Hi everyone! We’re team 8. Picture a drone the size of your palm — just 27 grams. The wind keeps pushing it, and it has to stay perfectly still in the air. We teach it to do that with Recurrent PPO — PPO with an LSTM memory. Let’s see how it works.

## 2. Task: hold a point while invisible forces push — Speaker A (~35 s)

The task sounds simple: hold a point. The drone knows where it is, how fast it’s moving and how it’s tilted — with noise and a small delay. What it doesn’t know is what’s pushing it: wind up to three and a half metres per second, its real mass, a weak motor. All of this is random in every episode. So the drone has to infer the disturbances from how it behaves. On the right is our simulator in a 3 m/s north wind.

## 3. Environment: one control step — Speaker A (~36 s)

Here’s one control step. Fifty times a second the network decides what attitude to take: which way to tilt, how to turn and how much thrust to give. It doesn’t drive the motors directly — an autopilot does that, ten times faster. That’s exactly how a real Crazyflie is controlled. In return the network gets 22 numbers about its state. The reward: closer and smoother is better, and a crash costs minus 50. We wrote the simulator ourselves — it’s fast, with wind, motor lag and delays.

## 4. Why memory: one frame can’t tell what’s pushing — Speaker A (~45 s)

So why does it need memory? Look: the drone is 3 centimetres below the target. Why? Is it heavier? Is the wind pushing it down? Or did it just wobble? From one frame you can’t tell. But if the error stays for a couple of seconds, there’s a constant force — add thrust. That’s a POMDP: the world is only partly observable. A classic PID handles this with a “piggy bank” — the integral of the error: it keeps adding up the error until it’s fixed. We want the network to learn that kind of memory by itself. Now my teammate will show how the algorithm works.

## 5. PPO: learning in small, careful steps — Speaker B (~38 s)

Thanks! Building block one — PPO. It has two networks. The actor is the pilot: it looks and picks an action. The critic is the coach: it estimates how much reward lies ahead. If things went better than the critic expected, we do that more often; if worse — less often. PPO’s key trick is on the plots: the policy isn’t allowed to change too sharply. The ratio of the new policy to the old one is clipped to a corridor from 0.8 to 1.2. That keeps training stable, without jumps.

## 6. LSTM: a network with a memory tape — Speaker B (~45 s)

Building block two — LSTM, a network made for time series. Data comes as a stream, fifty frames a second, and the past matters. The green line on top is long-term memory, the cell state: information flows along it almost untouched, so the network remembers seconds, not just a couple of frames. Three gates decide: what to forget — the wind has dropped; what to write — the error persists; and what to output right now. The network learns how to open them by itself. And note: our LSTM doesn’t predict the trajectory — it sits inside the policy and decides what from the past is useful.

## 7. RPPO = PPO + LSTM — Speaker B (~39 s)

Now put them together. RPPO is PPO with an LSTM in both the actor and the critic. Every step the network takes the new observation plus its memory and decides what to do. That’s how the drone gradually figures out the wind and its own weight. We train on chunks of 512 steps — ten seconds of flight — and push the error back through time: that’s BPTT. And one trick: during training the critic gets a cheat sheet — the true wind and mass. In flight it isn’t needed; only the actor flies.

## 8. How we train — Speaker B (~28 s)

Training: 16 drones fly in parallel, 3 million steps in total — about 17 hours of flight, or roughly forty minutes on an ordinary CPU. Every episode brings new random disturbances — that’s domain randomization. Difficulty grows gradually from calm to full — that’s a curriculum. And we keep the best version according to a separate validation set. Back to my teammate for the big picture.

## 9. Project idea: what does learned memory give? — Speaker A (~41 s)

So, the main question of the project: what does learned memory actually give us? We put four approaches side by side: PID with its hand-made piggy bank; PPO with no memory; PPO that gets the piggy bank as an input; and our RPPO, which learns its memory by itself. All of them fly in identical conditions. We measure accuracy, crashes and how they react to gusts. A first result is already in: RPPO holds the point with a median error of about 2.5 centimetres and not a single crash in validation. More in the next part. Thank you!

Total: 786 words ≈ 5.6 min (A: 428, B: 358).
