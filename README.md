<p align="center">
  <img src="https://outskirts.altervista.org/forum/ext/dmzx/imageupload/img-files/2/678e472/7575e5c/d73f48e81ad1cb59135c33ad3599cee5.png">
</p>

<h1 align="center">JudaS ++</h1>

## JudaS ++ Overview


JudaS ++ is a free and strong UCI chess engine derived from Stockfish 
that analyzes chess positions and computes the optimal moves.

JudaS ++ does not include a graphical user interface (GUI) that is required 
to display a chessboard and to make it easy to input moves. These GUIs are 
developed independently from JudaS and are available online.

## UCI options

  * #### Time Contempt
Type: Integer
Default Value: 20
Range: -100 to 100
Description:
This option sets the engine's "time contempt" factor, which influences how aggressively the engine utilizes its allocated time during the search. The value determines whether the engine should conserve time for future moves or spend more time searching for the best move at the current position.

Purpose:
Helps balance time management strategies based on the game situation.
Encourages aggressive play by spending more time in critical positions or conservative play by saving time for later.
How It Works:
Positive Values (e.g., 20):
The engine is more willing to spend time in earlier moves, searching deeper in critical positions.
Suitable for situations where early-game accuracy is critical.
Negative Values (e.g., -20):
The engine becomes more conservative with time usage, preserving resources for later stages of the game.
Useful in positions where time control or efficiency is essential.
Use Cases:
High Positive Values (e.g., 50 to 100):
Encourages deep analysis and aggressive play early in the game.
Ideal for long time controls or when playing for a win.
High Negative Values (e.g., -50 to -100):
Promotes cautious time management and efficiency.
Suitable for shorter time controls or drawish positions.

The following is a list of options supported by JudaS (on top of all UCI options supported by Stockfish)
  * #### CTG/BIN Book File
    The file name of the first book file which could be a polyglot (BIN) or Chessbase (CTG) book. To disable this book, use: ```<empty>```
    If the book (CTG or BIN) is in a different directory than the engine executable, then configure the full path of the book file, example:
    ```C:\Path\To\My\Book.ctg``` or ```/home/username/path/to/book/bin```

  * #### Book Width
    The number of moves to consider from the book for the same position. To play best book move, set this option to 1. If a value ```n``` (greater than 1 is configured, the engine will pick **randomly** one of the top ```n``` moves available in the book for the given position

  * #### Book Depth
    The maximum number of moves to play from the book
	
  * #### Self-Learning
	Experience file structure:

1. e4 (from start position)
1. c4 (from start position)
1. Nf3 (from start position)
1 .. c5 (after 1. e4)
1 .. d6 (after 1. e4)

2 positions and a total of 5 moves in those positions

Now imagine JudaS ++ plays 1. e4 again, it will store this move in the experience file, but it will be duplicate because 1. e4 is already stored. The experience file will now contain the following:
1. e4 (from start position)
1. c4 (from start position)
1. Nf3 (from start position)
1 .. c5 (after 1. e4)
1 .. d6 (after 1. e4)
1. e4 (from start position)

Now we have 2 positions, 6 moves, and 1 duplicate move (so effectively the total unique moves is 5)

Duplicate moves are a problem and should be removed by merging with existing moves. The merge operation will take the move with the highst depth and ignore the other ones. However, when the engine loads the experience file it will only merge duplicate moves in memory without saving the experience file (to make startup and loading experience file faster)

At this point, the experience file is considered fragmented because it contains duplicate moves. The fragmentation percentage is simply: (total duplicate moves) / (total unique moves) * 100
In this example we have a fragmentation level of: 1/6 * 100 = 16.67%

  * #### Experience Readonly
  Default: False If activated, the experience file is only read.
  
  * #### Experience Book
  SugaR play using the moves stored in the experience file as if it were a book

  * #### Experience Book Width
The number of moves to consider from the book for the same position.

  * #### Experience Book Eval Importance
The quality of experience book moves has been revised heavily based on feedback from users. The new logic relies on a new parameter called (Experience Book Eval Importance) which defines how much weight to assign to experience move evaluation vs. count.

The maximum value for this new parameter is: 10, which means the experience move quality will be 100% based on evaluation, and 0% based on count

The minimum value for this new parameter is: 0, which means the experience move quality will be 0% based on evaluation, and 100% based on count

The default value for this new parameter is: 5, which means the experience move quality will be 50% based on evaluation, and 50% based on count	

  * #### Experience Book Min Depth
Type: Integer
Default Value: 27
Range: Max to 64
This option sets the minimum depth threshold for moves to be included in the engine's Experience Book. Only moves that have been searched at least to the specified depth will be considered for inclusion.

  * #### Experience Book Max Moves
ype: Integer
Default Value: 16
Range: 1 to 100
	This is a setup to limit the number of moves that can be played by the experience book.
	If you configure 16, the engine will only play 16 moves (if available).

  * #### NNUE StrategyMaterialWeight
Type: Integer (Range: -12 to 12)
Default: 0

Adjusts the weight given to material balance in the evaluation function. Higher values emphasize material over other positional factors.  

  * #### NNUE StrategyPositionalWeight
Type: Integer (Range: -12 to 12)
Default: 0

 Adjusts the weight given to positional considerations in the evaluation function. Higher values emphasize positional strategy over material.

  * #### NNUE ManualWeights Adjustment
Type: Boolean (true or false)
Default: false
Enables or disables manual weight adjustments for the engine's evaluation function. When enabled, users can directly modify NNUE StrategyMaterialWeight and NNUE StrategyPositionalWeight.
Usage:
true: NNUE ManualWeights adjustments are enabled.
false: The engine uses dynamic weight calculations based on the game phase and position.

  * #### NNUE StrategyMaterialWeight
Adjusts the weight given to material balance in the evaluation function. Higher values emphasize material over other positional factors.

Type: Integer (Range: -12 to 12)
Default: 0

  * #### NNUE StrategyPositionalWeight
Adjusts the weight given to positional considerations in the evaluation function. Higher values emphasize positional strategy over material.

Type: Integer (Range: -12 to 12)
Default: 0


  * #### Exploration Parameters
Option: Exploration Factor

Type: Floating-point (Range: 0.0 to 3.0)
Default: 0.2
Description: Controls the balance between exploration and exploitation during the search. A higher value increases exploration.

  * #### Option: Dynamic Exploration

Type: Boolean (true or false)
Default: true
Description: Toggles dynamic adjustments for the exploration factor. When enabled, the engine adapts exploration levels based on time constraints and position complexity.


