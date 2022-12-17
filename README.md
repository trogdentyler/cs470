# CS 470 Semester Project

## ___________________________________________________________

I originally had decided to create a RISK agent for the semester project but soon realized, after looking at the 
provided code, that most of the ideas that I wanted to implement has already been done. So, in the end, I decided
to start from stratch and make my own chess playing agent.

To do this I used several guides online and borrowed code for specific sections of my project. Most of the code is
mine that I wrote following several psuedocode templates on [Chess Programming Wiki](https://www.chessprogramming.org/Main_Page), 
but the app portion of the code and ideas about opening moves come
from [Ansh Gaikwad](https://medium.com/dscvitpune/lets-create-a-chess-ai-8542a12afef).

My goal was to create a chess playing agent using a NegaMax search with Alpha-Beta
pruning. I did this using a common heuristic found in the psuedocode. This 
heuristic involves calculating the material and positioning of one's pieces
on the board. I later wanted to implement a deep learning heuristic which 
essentially mimics Stockfish. For this I borrowed code from several sources but
in the end it did not work, so there's future work to be done there.

After first having created the agent to make moves, it was aparent that it
was taking an extremely long time to pick a move; therfore, I also implmented 
a transition table to keep track of search for moves. This definelty sped up 
the agent so that it plays in a reasonable amount of time. However, I am still
only able to search about 4 moves deep, which at times gives poor results.
I also added Iterative Deepening in hopes that that agent would make better
moves. It's hard to say whether or not the moves are better but it did slow
the agent down some.

Also, note that there is a bug if you repeatedly press the "Make AI Move" 
button when the agent is still calculating a move to make. So in order to play 
a full game without any glitches you cannot rapidly press buttons.