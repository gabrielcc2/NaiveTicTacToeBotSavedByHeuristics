import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
//import java.lang.Math;
import de.ovgu.dke.teaching.ml.tictactoe.api.IBoard;
import de.ovgu.dke.teaching.ml.tictactoe.api.IPlayer;
import de.ovgu.dke.teaching.ml.tictactoe.api.IllegalMoveException;
import de.ovgu.dke.teaching.ml.tictactoe.game.Move;
/**
 * @author 
 * 		1) Gabriel Campero
 * 		2) Rene Tatua Castillo
 * 		3) Vishnu Unnikrishnan
 * 
 ****************Quick overview of code****************
 * Here we present a tic-tac-toe player based on reinforcement learning, 
 * modeled similar to linear regression. 
 * 
 * It takes as an input a file called weightsTequilaBot.txt, of 125 lines, each with 9 csv.
 * 
 * Since boards at different stages of the game cannot be compared correctly, 
 * the model aims to have a set of weights for every stage of the game.
 * 
 * This is done by having a set of weights W0-W8 for every stage/turn/move of the game
 * (move 1, move 2,...... move 125 etc.). From 1 to 125 possible moves.
 * 
 * Each set of W0 to W8 weights represent, for a given turn, respectively: 
 *  W0: Number of unblocked lines with 4 chips of our player.
 *  W1: Number of unblocked lines with 3 chips of our player.
 *  W2: Number of unblocked lines with 2 chips of our player.
 *  W3: Number of unblocked lines with 1 chips of our player.
 *  W4: Number of unblocked lines with 4 chips of the opponent.
 *  W5: Number of unblocked lines with 3 chips of our opponent.
 *  W6: Number of unblocked lines with 2 chips of our opponent.
 *  W7: Number of unblocked lines with 1 chip of our opponent.
 *  W8: Independent variable
 *  
 *  We store the learned weights for every move (125 rows x 9 columns)
 * in a weightsTequilaBot.txt file, which is loaded and saved respectively at endgame / beginning.
 * 
 *  The appropriate loaded weights are used to score the board after a given play and learn from result (win/loss/draw)
 *  at every stage of the game, and is also stored back into the file on endgame. 
 *  
 *  Also we test the score of the board after all potential moves, and from this process we decide which move is
 *  the best move at this stage of the game, according to our variables.
 *  
 *  Given that this model is perhaps not the most suited, according to literature, for this problem, we
 *  tried different optimizations, studying how much this model could be improved by them. Here are some of them:
 *  
 *  1) Our model has certain heuristics implemented, for instance, when an opportunity of winning with 1 move is spotted
 *  the scoring function is not followed, and instead the winningPosition is given.
 *  Similarly, when the opponent could win with only one move, we decide to block the learning function and instead
 *  use a heuristic to play so as to block the opponent.
 *  
 *  A final heuristic consists on always selecting for the first move (if there is chance for it), the center of the board.
 *  
 *  The integration of these heuristics with the weight update functions was something that we thought could affect negatively our model, since the score of the board from those moves does not represent really the tendency of the playing according to the scoring. Because of this reasoning we decided not to learn from the board at those moves.
 *  In this way, our model is only learning the path towards winning states, but not the winning move itself. It is also learning which paths help to avoid a loss, but not the specific move to avoid it.
 *  
 *  2) An additional aspect we considered was depth scaling. So as to reward a set of moves that leads to a quicker win, we decided to score
 *  each move from this game (apart from the heuristic-driven moves, of course) by: 125-numberOfFinalMovesOfTheGame. And correspondingly,
 *  by numberOfFinalMoves-125 for the case of loses.
 *  
 *  3) Since we considered that we wanted to reward higher the lines with more moves, we decided to initialize our model with
 *  W0=8, W1=4, W2=2, W3=1, W4=-8, W5=-4, W6=-2, W7=-1, W8=1. This was decided in a trial and error way, for we did not have a clear
 *  framework for evaluating the alternatives.
 *  
 *  Given these weights we trained our model in 1 tournament of 10 rounds against RandomPlayer:SmartPlayer:SmartPlayer.
 *  Perhaps more training would be helpful, but given the time required for it, we had to settle for this.
 *  
 *  4) Since we noticed that our model might to be biased towards draws (from the heuristics used)
 *  we decided that during the training of our model we would not allow learning for draws, but only for wins and loses.
 *  
 *  This class implements the following public functions:
 *  public String getName(): returns name of the player
 *  
 *  Inner private functions:
 *  private int[][] boardToLineArray (IBoard ): Changes a board to an array of 109 lines.
 *  private int[] findFeatures(int[][] lines): Given the former array, calculates the X0-X7 variables or features.
 *  private boolean imminentVictory(IBoard ): Asserts if there is a chance for winning in this move.
 *  private boolean imminentDefeat(IBoard ): Asserts if there is a chance of the opponent winning in the next move. 
 *  private void loadExperience(): Loads the experience or weights.
 *  private double score(int []): Scores the board, according to a set of variables from findFeatures, 
 *                                   and the weights for the current move, signaled by the variable turn.
 *  private int[] maximumScorePosition(): Selects the position with a maximum score, according to our calculation in the current turn.
 *  private void addAndStoreExperience(IBoard ): Stores the experience of a given final board, updating the weights and writing to the file.
 *  private int[] selectMove (IBoard ): Used by makeMove, selects the best move given a board, using our calculation and no heuristics.
 *  public int[] makeMove(IBoard ): The logic of making a move, it checks if variables need initialization, then if
 *                                       heuristics apply, finally if not, it calls selectMove.
 *  public void onMatchEnds(IBoard ): What is done when the match ends.
 */

public class tequilaBot implements IPlayer {
	/*Set of global variables describing the model*/
	double thetas[][]=new double [125][9]; //The stored thetas or weights.
	double learningRate=0.1;
	
	/*Variables with information about the board*/
	int posCount=125; //The number of positions in a 5*5*5 board.
	int dimSize=5;
	
	/*Helper variables used in different methods*/
	boolean experienceLoaded=false;	
	double scoresboard[][][] = new double[5][5][5]; //Board holding the scores of tentative moves.
	int winningPos[]=new int[3]; //In the case of a imminent victory, here will be the decisive move.
	int panickedMove[]=new int[3]; //In the case of a possible defeat, here will be the move that could counter that chance.
	int turn=0; //Global variable that must be tracked. It is used to convey the current move to the scoring function, which, otherwise would lack access to that information.
	boolean tentative=true; //Flag defining the tentativeness of a move.
	boolean winCheck=true;  //Flag
	boolean learnFromThisMove[]= new boolean [125]; //Flags so the model doesn't learn from moves done by heuristics.
	
	
	//Private functions

	//Private inner functions
	/*Function: boardToLineArray
	 * arguments: IBoard copy
	 * returns: Lines array with 109 lines and 5 points for each line.
	 * 
	 * 	What the function does:
	 * 	Each board has a fixed number of winning lines in it. In the case of a 5x5x5 board,
	 * 	there are 109 distinct winning lines in the game. Each move that you / the opponent makes changes one of these lines.
	 * 
	 *  This function scans all the pieces on the board, and traverses a plane for each horizontal, vertical layer, and
	 *  also the layers for each depth.
	 *  Horizontal layers: traverse x,y for each layer in z (12x5 = 60 unique lines)
	 *  Vertical layers: traverse x,z for each layer in y	(7x5 = 35 unique lines)
	 *  Depth layers: traverse y,z for each layer in x.		(2x5 = 10 unique lines)
	 *  In addition to this, there are four main 3D diagonals that connect the opposite vertices of the cube
	 *  3D Diagonals										(4x1 =	4 unique lines)
	 *	This totals to 60+35+10+4 = 109 diagonals.  
	 *
	 *  For line in every one of these layers, the positions inside 
	 *  the line array indicate whether a move by the player, the 
	 *  opponent or an empty space was encountered. Counters keep 
	 *  track of each of these possibilities for easier handling of cases. 
	 *   
	 */
	private int[][] boardToLineArray (IBoard copy){
		int thisGameLines [][]=new int [109][5];	//Stores the 109 lines
		int currLine=0; 							//Maintains indexing into thisGameLines[][]

		
		//Layering the cake
		int numElems=0;			//Tracks no. of pieces of player that was found.
		int numOp=0;			//Tracks no. of pieces of opponent that was found.
		for (int layer=0; layer<5; layer++){
			//To find horizontal lines of layer
			for (int i=0; i<dimSize; i++){
				numElems=0; 	//Reset count for player pieces for each new line
				numOp=0;		//Reset count for opponent pieces for each new line
				for (int j=0; j<dimSize; j++){
					int extraVal[]=new int [] {layer,i,j};
					if (copy.getFieldValue(extraVal)==null) {
						thisGameLines[currLine][j]=0;
					}
					else if (copy.getFieldValue(extraVal).toString().contains("tequilaBot")){
						thisGameLines[currLine][j]=1;
						numElems++;//Found move by player
					}
					else {
						thisGameLines[currLine][j]=-1;
						numOp++;//Found move by opponent
					}
				}
				//For tentative flag, refer documentation in 'makeMove()' winCheck is to reduce work, so we only check here when needed.
				if(numElems>=4 && numOp==0 && !tentative && winCheck){//You have four 'x's and one blank in the line
					for (int l=0; l<5; l++){
						int extraVal[]=new int [] {layer,i,l};
						if (copy.getFieldValue(extraVal)==null) {
							winningPos=extraVal;
							//Find the one blank in the line, and return it as the next move
						}
					}
				}
				else if (numOp>=4 && numElems==0 && !tentative && !winCheck){//Your opponent has four 'o's and one blank in the line
					for (int l=0; l<5; l++){
						int extraVal[]=new int [] {layer,i,l};
						if (copy.getFieldValue(extraVal)==null) {
							panickedMove=extraVal;
						}
					}
				}
				currLine++;
			}			
			//Vertical lines of layer
			for (int j=0; j<dimSize; j++){
				numElems=0; 
				numOp=0;
				for (int i=0; i<dimSize; i++){
					int extraVal[]=new int [] {layer,i,j};
					if (copy.getFieldValue(extraVal)==null) {
						thisGameLines[currLine][i]=0;
					}
					else if (copy.getFieldValue(extraVal).toString().contains("tequilaBot")){
						thisGameLines[currLine][i]=1;
						numElems++;
					}
					else {
						thisGameLines[currLine][i]=-1;
						numOp++;
					}
				}
				if(numElems>=4 && numOp==0 && !tentative && winCheck){
					for (int l=0; l<5; l++){
						int extraVal[]=new int [] {layer,l,j};
						if (copy.getFieldValue(extraVal)==null) {
							winningPos=extraVal;
						}
					}
				}
				else if (numOp>=4 && numElems==0 && !tentative && !winCheck){
					for (int l=0; l<5; l++){
						int extraVal[]=new int [] {layer,l,j};
						if (copy.getFieldValue(extraVal)==null) {
							panickedMove=extraVal;
						}
					}
				}
				currLine++;
			}
			//First diagonal line of layer From 00 to 44
			
			numElems=0; 
			numOp=0;
			for (int i=0;i<5;i++){
				int extraVal[]=new int [] {layer,i,i};
				if (copy.getFieldValue(extraVal)==null) {
					thisGameLines[currLine][i]=0;
				}
				else if (copy.getFieldValue(extraVal).toString().contains("tequilaBot")){
					thisGameLines[currLine][i]=1;
					numElems++;
				}
				else {
					thisGameLines[currLine][i]=-1;
					numOp++;
				}
			}
			if(numElems>=4 && numOp==0 && !tentative && winCheck){
				for (int l=0; l<5; l++){
					int extraVal[]=new int [] {layer,l,l};
					if (copy.getFieldValue(extraVal)==null) {
						winningPos=extraVal;
					}
				}
			}
			else if (numOp>=4 && numElems==0 && !tentative && !winCheck){
				for (int l=0; l<5; l++){
					int extraVal[]=new int [] {layer,l,l};
					if (copy.getFieldValue(extraVal)==null) {
						panickedMove=extraVal;
					}
				}
			}
			currLine++;
		
			//Second diagonal line of layer From 04 to 40
			numElems=0; 
			numOp=0;
			for (int i=0;i<5;i++){
				int extraVal[]=new int [] {layer,i,4-i};
					if (copy.getFieldValue(extraVal)==null) {
						thisGameLines[currLine][i]=0;
					}
					else if (copy.getFieldValue(extraVal).toString().contains("tequilaBot")){
						thisGameLines[currLine][i]=1;
						numElems++;
					}
					else {
						thisGameLines[currLine][i]=-1;
						numOp++;
					}
			}
			if(numElems>=4 && numOp==0 && !tentative && winCheck){
				for (int l=0; l<5; l++){
					int extraVal[]=new int [] {layer,l,4-l};
					if (copy.getFieldValue(extraVal)==null) {
						winningPos=extraVal;
					}
				}
			}
			else if (numOp>=4 && numElems==0 && !tentative && !winCheck){
				for (int l=0; l<5; l++){
					int extraVal[]=new int [] {layer,l,4-l};
					if (copy.getFieldValue(extraVal)==null) {
						panickedMove=extraVal;
					}
				}
			}
			currLine++;
			numOp=0;
			numElems=0;
	}
		//We will aim at 35 lines for this.
		//At this point we have 60 lines of 109. 49 to go.		
		//Cutting the cake
		
		for (int layer=0; layer<5; layer++){
			//Vertical lines of layer
			for (int j=0; j<dimSize; j++){
				numElems=0; 
				numOp=0;
				for (int k=0; k<dimSize; k++){
					int extraVal[]=new int [] {k,layer,j};
					if (copy.getFieldValue(extraVal)==null) {
						thisGameLines[currLine][k]=0;
					}
					else if (copy.getFieldValue(extraVal).toString().contains("tequilaBot")){
						thisGameLines[currLine][k]=1;
						numElems++;
					}
					else {
						thisGameLines[currLine][k]=-1;
						numOp++;
					}
				}
				if(numElems>=4 && numOp==0 && !tentative && winCheck){
					for (int l=0; l<5; l++){
						int extraVal[]=new int [] {l,layer,j};
						if (copy.getFieldValue(extraVal)==null) {
							winningPos=extraVal;
						}
					}
				}
				else if (numOp>=4 && numElems==0 && !tentative && !winCheck){
					for (int l=0; l<5; l++){
						int extraVal[]=new int [] {l,layer,j};
						if (copy.getFieldValue(extraVal)==null) {
							panickedMove=extraVal;
						}
					}
				}
				currLine++;
			}
			//First diagonal line of layer From 00 to 44
				numElems=0; 
				numOp=0;
			for (int i=0;i<5;i++){
				int extraVal[]=new int [] {i, layer, i};
				if (copy.getFieldValue(extraVal)==null) {
					thisGameLines[currLine][i]=0;
				}
				else if (copy.getFieldValue(extraVal).toString().contains("tequilaBot")){
					thisGameLines[currLine][i]=1;
					numElems++;
				}
				else {
					thisGameLines[currLine][i]=-1;
					numOp++;
				}
			}
			if(numElems>=4 && numOp==0 && !tentative && winCheck){
				for (int l=0; l<5; l++){
					int extraVal[]=new int [] {l, layer,l};
					if (copy.getFieldValue(extraVal)==null) {
						winningPos=extraVal;
					}
				}
			}
			else if (numOp>=4 && numElems==0 && !tentative && !winCheck){
				for (int l=0; l<5; l++){
					int extraVal[]=new int [] {l,layer,l};
					if (copy.getFieldValue(extraVal)==null) {
						panickedMove=extraVal;
					}
				}
			}

			currLine++;
		
			//Second diagonal line of layer From 04 to 40
			numElems=0; 
			numOp=0;
			for (int i=0;i<5;i++){
				int extraVal[]=new int [] {i,layer,4-i};
				if (copy.getFieldValue(extraVal)==null) {
					thisGameLines[currLine][i]=0;
				}
				else if (copy.getFieldValue(extraVal).toString().contains("tequilaBot")){
						thisGameLines[currLine][i]=1;
						numElems++;
					}
				else {
						thisGameLines[currLine][i]=-1;
						numOp++;
					}
			}
			if(numElems>=4 && numOp==0 && !tentative && winCheck){
				for (int l=0; l<5; l++){
					int extraVal[]=new int [] {l,layer,4-l};
					if (copy.getFieldValue(extraVal)==null) {
						winningPos=extraVal;
					}
				}
			}
			else if (numOp>=4 && numElems==0 && !tentative && !winCheck){
				for (int l=0; l<5; l++){
					int extraVal[]=new int [] {l,layer,4-l};
					if (copy.getFieldValue(extraVal)==null) {
						panickedMove=extraVal;
					}
				}
			}
			currLine++;
		}

		//Cutting the cake in a direction perpendicular to the previous.
		for (int layer=0; layer<5; layer++){
			//First diagonal line of layer From 00 to 44
			numElems=0; 
			numOp=0;
			for (int i=0;i<5;i++){
				int extraVal[]=new int [] {i, i, layer};
				if (copy.getFieldValue(extraVal)==null) {
					thisGameLines[currLine][i]=0;
				}
				else if (copy.getFieldValue(extraVal).toString().contains("tequilaBot")){
					thisGameLines[currLine][i]=1;
					numElems++;
				}
				else {
					thisGameLines[currLine][i]=-1;
					numOp++;
				}
			}
			if(numElems>=4 && numOp==0 && !tentative && winCheck){
				for (int l=0; l<5; l++){
					int extraVal[]=new int [] {l, l, layer};
					if (copy.getFieldValue(extraVal)==null) {
						winningPos=extraVal;
					}
				}
			}
			else if (numOp>=4 && numElems==0 && !tentative && !winCheck){
				for (int l=0; l<5; l++){
					int extraVal[]=new int [] {l,l,layer};
					if (copy.getFieldValue(extraVal)==null) {
						panickedMove=extraVal;
					}
				}
			}
			currLine++;
						
			numElems=0; 
			numOp=0;
			//Second diagonal line of layer From 04 to 40
			for (int i=0;i<5;i++){
				int extraVal[]=new int [] {i,4-i, layer};
					if (copy.getFieldValue(extraVal)==null) {
					thisGameLines[currLine][i]=0;
					}
				else if (copy.getFieldValue(extraVal).toString().contains("tequilaBot")){
						thisGameLines[currLine][i]=1;
						numElems++;
					}
				else {
						thisGameLines[currLine][i]=-1;
						numOp++;
					}
			}
			if(numElems>=4 && numOp==0 && !tentative && winCheck){
				for (int l=0; l<5; l++){
					int extraVal[]=new int [] {l, 4-l, layer};
					if (copy.getFieldValue(extraVal)==null) {
						winningPos=extraVal;
					}
				}
			}
			else if (numOp>=4 && numElems==0 && !tentative && !winCheck){
				for (int l=0; l<5; l++){
					int extraVal[]=new int [] {l,4-l,layer};
					if (copy.getFieldValue(extraVal)==null) {
						panickedMove=extraVal;
					}
				}
			}
			currLine++;
		}
		
		//And now the 4 3D diagonals, going from corner to corner of the board.
		
		numElems=0; 
		numOp=0;
		for (int i=0;i<5;i++){
			int extraVal[]=new int [] {i, i, i};
			if (copy.getFieldValue(extraVal)==null) {
				thisGameLines[currLine][i]=0;
			}
			else if (copy.getFieldValue(extraVal).toString().contains("tequilaBot")){
				thisGameLines[currLine][i]=1;
				numElems++;
			}
			else {
				thisGameLines[currLine][i]=-1;
				numOp++;
			}
		}
		if(numElems>=4 && numOp==0 && !tentative && winCheck){
			for (int l=0; l<5; l++){
				int extraVal[]=new int [] {l, l, l};
				if (copy.getFieldValue(extraVal)==null) {
					winningPos=extraVal;
				}
			}
		}
		else if (numOp>=4 && numElems==0 && !tentative && !winCheck){
			for (int l=0; l<5; l++){
				int extraVal[]=new int [] {l,l,l};
				if (copy.getFieldValue(extraVal)==null) {
					panickedMove=extraVal;
				}
			}
		}
		currLine++;
		numElems=0; 
		numOp=0;
		for (int i=0;i<5;i++){
			int extraVal[]=new int [] {i,i,4-i};
			if (copy.getFieldValue(extraVal)==null) {
				thisGameLines[currLine][i]=0;
			}
			else if (copy.getFieldValue(extraVal).toString().contains("tequilaBot")){
					thisGameLines[currLine][i]=1;
					numElems++;
				}
			else {
					thisGameLines[currLine][i]=-1;
					numOp++;
				}
		}
		if(numElems>=4 && numOp==0 && !tentative && winCheck){
			for (int l=0; l<5; l++){
				int extraVal[]=new int [] {l, l, 4-l};
				if (copy.getFieldValue(extraVal)==null) {
					winningPos=extraVal;
				}
			}
		}
		else if (numOp>=4 && numElems==0 && !tentative && !winCheck){
			for (int l=0; l<5; l++){
				int extraVal[]=new int [] {l,l,4-l};
				if (copy.getFieldValue(extraVal)==null) {
					panickedMove=extraVal;
				}
			}
		}
		currLine++;
		numElems=0; 
		numOp=0;
		for (int i=0;i<5;i++){
			int extraVal[]=new int [] {4-i,i,i};
			if (copy.getFieldValue(extraVal)==null) {
				thisGameLines[currLine][i]=0;
			}
			else if (copy.getFieldValue(extraVal).toString().contains("tequilaBot")){
					thisGameLines[currLine][i]=1;
					numElems++;
				}
			else {
					thisGameLines[currLine][i]=-1;
					numOp++;
				}
		}
		if(numElems>=4 && numOp==0 && !tentative && winCheck){
			for (int l=0; l<5; l++){
				int extraVal[]=new int [] {4-l, l, l};
				if (copy.getFieldValue(extraVal)==null) {
					winningPos=extraVal;
				}
			}
		}
		else if (numOp>=4 && numElems==0 && !tentative && !winCheck){
			for (int l=0; l<5; l++){
				int extraVal[]=new int [] {4-l,l,l};
				if (copy.getFieldValue(extraVal)==null) {
					panickedMove=extraVal;
				}
			}
		}
		currLine++;
		numElems=0; 
		numOp=0;
		for (int i=0;i<5;i++){
			int extraVal[]=new int [] {i,4-i,i};
			if (copy.getFieldValue(extraVal)==null) {
				thisGameLines[currLine][i]=0;
			}
			else if (copy.getFieldValue(extraVal).toString().contains("tequilaBot")){
					thisGameLines[currLine][i]=1;
					numElems++;
				}
			else {
					thisGameLines[currLine][i]=-1;
					numOp++;
				}
		}
				if(numElems>=4 && numOp==0 && !tentative && winCheck){
			for (int l=0; l<5; l++){
				int extraVal[]=new int [] {l, 4-l, l};
				if (copy.getFieldValue(extraVal)==null) {
					winningPos=extraVal;
				}
			}
		}
		else if (numOp>=4 && numElems==0 && !tentative && !winCheck){
			for (int l=0; l<5; l++){
				int extraVal[]=new int [] {l,4-l,l};
				if (copy.getFieldValue(extraVal)==null) {
					panickedMove=extraVal;
				}
			}
		}
		currLine++;				
	return thisGameLines;		
	}	
	
    private int[] findFeatures(int[][] lines){
    	int stats[]=new int [8]; //Array 0-3> Number of pos where we have 4,3,2,1 and a winning chance. Array from 4-7>Number of pos where opponent has 4,3,2,1 and a winning chance. Array 8, free lines, Array 9, blocked lines.
    	for (int i=0; i<8; i++){
    		stats[i]=0;
    	}
    	int opponentMoves, myMoves;
    	for (int i=0; i<109; i++){
    		opponentMoves=0;
    		myMoves=0;
    		for (int j=0; j<5;j++){
    			if (lines[i][j]==1){
    				myMoves++;
    			}
    			else if(lines[i][j]==-1){
    				opponentMoves++;
    			}
    		}
    		if (myMoves>0&&myMoves<5 && opponentMoves==0){
    			stats[4-myMoves]++;
    		}
    		else if (myMoves>0&&myMoves>=5 && opponentMoves==0){
    			stats[0]++;
    		}
    		else if(opponentMoves>0&&opponentMoves<5 && myMoves==0){
    			stats[8-opponentMoves]++;
    		}
    		else if(opponentMoves>0&&opponentMoves>=5 && myMoves==0){
    			stats[4]++;
    		}
    	}
    	return stats;
    }
  
    
    private boolean imminentVictory(IBoard board){
    	if(findFeatures(boardToLineArray(board))[0]>0){
    		return true;
    	}
    	return false;
    }
    
    
    private boolean imminentDefeat(IBoard board){
    	if(findFeatures(boardToLineArray(board))[4]>0){
    		return true;
    	}
    	return false;
    }

    
    //Loads the experience form the file and initializes certain supporting variables and flags.
    private void loadExperience(){
		//Initializations
    	turn=0;
		for (int i=0; i<125; i++){
			learnFromThisMove[i]=true; 
		}
		//And now the reading from the file...
		BufferedReader reader=null;  
		try {
			reader = new BufferedReader(new FileReader("weightsTequilaBot.txt"));
			String line = null;
			int i=0;
			int j=0; 
			while ((line = reader.readLine()) !=null)   {		    
						String[] splited = line.split(",");
						  for (String part : splited) {	
								  thetas[i][j]=Double.parseDouble(part);
								  j=j+1;
						  }
				i=i+1;
				j=0;
			}
		}
			catch (IOException e) {
			    e.printStackTrace();
			} finally {
			    try {
			        reader.close();
			    } catch (IOException e) {
			        e.printStackTrace();
			    }
			}
		}

	//This function scores the board in each stage or turn. 
    //Scores the board, according to a set of variables from findFeatures, and the weights for the current move, signaled by the variable turn.
 	private double score(int var[]){
 		double score_result=0; 
 		score_result=((thetas[turn][0]*var[0]+thetas[turn][1]*var[1]+thetas[turn][2]*var[2]+thetas[turn][3]*var[3])/(thetas[turn][4]*var[4]
 				+thetas[turn][5]*var[5]+thetas[turn][6]*var[6]+thetas[turn][7]*var[7]))+thetas[turn][8];
 		//score_result=thetas[turn][0]*var[0]+thetas[turn][1]*var[1]+thetas[turn][2]*var[2]+thetas[turn][3]*var[3]+thetas[turn][4]*var[4]
			//	+thetas[turn][5]*var[5]+thetas[turn][6]*var[6]+thetas[turn][7]*var[7]+thetas[turn][8];
 		return score_result;
	}
 	
 	//Returns the function with  the maximum score, according to our calculation.
	private int[] maximumScorePosition(){        
		int pos1=0;
		int pos2=0;
		int pos3=0;
		double maxScore=scoresboard[0][0][0]; //TODO There might be a bug from time to time, when the bot returns 0,0,0. The fix might be around here.
		for (int k=0; k<dimSize; k++){
			for (int i=0; i<dimSize; i++){
				for (int j=0; j<dimSize; j++){
						if (maxScore<scoresboard[k][i][j]){
					    	pos1=k;
					    	pos2=i;
					    	pos3=j;
					    	maxScore=scoresboard[k][i][j];
					    }
				}
			}
		}
		return new int []{pos1,pos2,pos3};
	}
 	
	//Adds and stores the experience of the game.
	private void addAndStoreExperience(IBoard copy){
		PrintWriter writer;
		int turns=copy.getMoveHistory().size(); //It was -1, because we previously did not learn from final boards. Now we do.
		int offset;
		IBoard replay=copy.clone();
		replay.clear();
		int y=(125-turns);
		if (copy.getWinner()!=null){
			if (copy.getWinner().getName()!=this.getName()){
				y=(turns-125);
			}
		}
		else{
			y=0;
		}
		if (copy.getMoveHistory().get(0).getPlayer().getName()==this.getName()){
			offset=0;
			try {
				replay.makeMove(copy.getMoveHistory().get(0));
			} catch (IllegalMoveException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		else {
			offset=1;
			try {
				replay.makeMove(copy.getMoveHistory().get(0));
			} catch (IllegalMoveException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			try {
				replay.makeMove(copy.getMoveHistory().get(1));
			} catch (IllegalMoveException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		/*First we recreate the game, updating thetas with winning knowledge...
		 * 
		 * *
		 */
		//if (y!=0){ //Since our learner seems biased against draws, we experimented with not learning from them.
			for (int i=offset; i<turns-1; i=i+2){
				try {
					replay.makeMove(copy.getMoveHistory().get(i+1));
				} catch (IllegalMoveException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				int vars[]=findFeatures(boardToLineArray(replay));
				turn=i;
				double tempScore=score(vars);
				double error=y-tempScore;
				//error=java.lang.Math.sqrt(error*error);
				if(learnFromThisMove[i]){
					System.out.println("Storing thetas for: "+ this.getName()+" Turn:"+i + " TempScore: "+tempScore+" y: "+y +" Error: "+error+" "+vars[0]+" "+vars[1]+" "+vars[2]+" "+vars[3]+" Op: "+vars[4]+" "+vars[5]+" "+vars[6]+" "+vars[7]);
					thetas[i][0]=thetas[i][0]+learningRate*vars[0]*error;
					thetas[i][1]=thetas[i][1]+learningRate*vars[1]*error;
					thetas[i][2]=thetas[i][2]+learningRate*vars[2]*error;
					thetas[i][3]=thetas[i][3]+learningRate*vars[3]*error;
					
					thetas[i][4]=thetas[i][4]+learningRate*vars[4]*error;
					thetas[i][5]=thetas[i][5]+learningRate*vars[5]*error;
					thetas[i][6]=thetas[i][6]+learningRate*vars[6]*error;
					thetas[i][7]=thetas[i][7]+learningRate*vars[7]*error;
					
					thetas[i][8]=thetas[i][8]+learningRate*error;
				}
				if (i+2<turns){
					try {
						replay.makeMove(copy.getMoveHistory().get(i+2));
					} catch (IllegalMoveException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					} 	
				}
			}
		//} //
		try {
			writer = new PrintWriter("weightsTequilaBot.txt", "UTF-8");
			for (int i=0;i<125;i++){
				writer.println(thetas[i][0] + ","
						+thetas[i][1] + ","
						+thetas[i][2] + ","
						+thetas[i][3] + ","
						+thetas[i][4] + ","
						+thetas[i][5] + ","
						+thetas[i][6] + ","
						+thetas[i][7] + ","
						+thetas[i][8]);
			}
			writer.close();
		} catch (FileNotFoundException | UnsupportedEncodingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}	
	}
	
	private int[] selectMove (IBoard board){
		turn=board.getMoveHistory().size();
		//First we clean the scoresboard:
		for (int k=0; k<dimSize; k++){
			for (int i=0; i<dimSize; i++){
				for (int j=0; j<dimSize; j++){
					scoresboard[k][i][j]=0;
				}
			}
		 }
		//Now we interate on all positions and calculate the score of the board if they were used to make a move..
		 for (int k=0; k<dimSize; k++){
			for (int i=0; i<dimSize; i++){
				for (int j=0; j<dimSize; j++){
					int extraVal[]=new int [] {k,i,j};
					if (board.getFieldValue(extraVal)!=null){
						scoresboard[k][i][j]=Double.NEGATIVE_INFINITY;
					}
					else {
						 IBoard copy=board.clone();
						 try {
								copy.makeMove(new Move(this, extraVal));
								scoresboard[k][i][j]=score(findFeatures(boardToLineArray(copy)));
							} catch (IllegalMoveException e) {
								// move was not allowed
							}
				     }
				}
			}
       }
	  int returnVal[]=maximumScorePosition();
//Useful for debugging:	  System.out.println("maxScore: "+returnVal[0]+" "+returnVal[1]+" "+returnVal[2]+" "+scoresboard[returnVal[0]][returnVal[1]][returnVal[2]]);
      return returnVal;
	}
	
	//Public functions
	public String getName() {
		// TODO Auto-generated method stub
		return "TequilaBot";
	}
	
	public int[] makeMove(IBoard board) {
		// TODO Auto-generated method stub
		if (!experienceLoaded)
		{
			this.loadExperience();
			experienceLoaded=true;
			
		}
		if(board.getMoveHistory().size()==0)
		{
//Useful for debugging:			System.out.println("First Move detected - forcing 2,2,2)");
			learnFromThisMove[0]=false; //Since we will move by heuristics, we dont learn for this move.
			return new int[] {2,2,2};
		}
		// do a move using the cloned board
		tentative=false; winCheck=true;
		boolean willWin=imminentVictory(board);
		tentative=true;	winCheck=false;
		int[] tentativeMove=null;
		if (willWin){
			tentativeMove=winningPos;
			learnFromThisMove[board.getMoveHistory().size()]=false;
			return winningPos;
		}
	/*	else{
			winCheck=false; tentative=false;
			boolean mightLose=imminentDefeat(board);
			tentative=true;
			if (mightLose){
				tentativeMove=panickedMove;
				learnFromThisMove[board.getMoveHistory().size()]=false;
				return panickedMove;
			}
		}*/
		tentativeMove=selectMove(board);
		return tentativeMove;
	}

	
	public void onMatchEnds(IBoard board) {
		this.addAndStoreExperience(board); 
		//The previous line can be commented so the bot doesn't learn during the tournament.
		return;
	}
}
