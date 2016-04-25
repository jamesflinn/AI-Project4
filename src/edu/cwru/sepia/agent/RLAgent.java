package edu.cwru.sepia.agent;

import com.sun.glass.ui.EventLoop;
import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.action.ActionFeedback;
import edu.cwru.sepia.action.ActionResult;
import edu.cwru.sepia.action.TargetedAction;
import edu.cwru.sepia.environment.model.history.DamageLog;
import edu.cwru.sepia.environment.model.history.DeathLog;
import edu.cwru.sepia.environment.model.history.History;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.Unit;
import edu.cwru.sepia.environment.model.state.Unit.UnitView;

import java.io.*;
import java.util.*;

public class RLAgent extends Agent {

    /**
     * Set in the constructor. Defines how many learning episodes your agent should run for.
     * When starting an episode. If the count is greater than this value print a message
     * and call sys.exit(0)
     */
    public final int numEpisodes;

    private int currentEpisode;
    private int episodesTested;
    private int episodesEvaluated;
    private boolean testingEpisode;
    private int episodesWon;

    /**
     * List of your footmen and your enemies footmen
     */
    private List<Integer> myFootmen;
    private List<Integer> enemyFootmen;
    private List<Action> currentActions;

    /**
     * Convenience variable specifying enemy agent number. Use this whenever referring
     * to the enemy agent. We will make sure it is set to the proper number when testing your code.
     */
    public static final int ENEMY_PLAYERNUM = 1;

    /**
     * Set this to whatever size your feature vector is.
     */
    public static final int NUM_FEATURES = 6;
    public static final int NUM_ATTACKING_FOOTMEN_FEATURE = 1;
    public static final int BEING_ATTACKED_FEATURE = 2;
    public static final int CLOSEST_ENEMY_FEATURE = 3;
    public static final int HEALTH_FEATURE = 4;
    public static final int WEAKEST_ENEMY_FEATURE = 5;

    /** Use this random number generator for your epsilon exploration. When you submit we will
     * change this seed so make sure that your agent works for more than the default seed.
     */
    public final Random random = new Random(12345);

    /**
     * Your Q-function weights.
     */
    public double[] weights;
    private double[] bestWeights;
    private String[] featureNames = {"constant", "footmen attacking", "being attacked", "closest enemy",
                                     "health", "weakest enemy", "friendlies", "enemies"};

    // Rewards
    private List<Double> allRewards        = new ArrayList<>();
    private List<Double> currentRewards    = new ArrayList<>();
    private List<Double> averageRewards    = new ArrayList<>();
    private List<Double> evaluationRewards = new ArrayList<>();
    private double bestReward = 0;

    private Map<Integer, Double[]> previousFeatureValues = new HashMap<>();

    /**
     * These variables are set for you according to the assignment definition. You can change them,
     * but it is not recommended. If you do change them please let us know and explain your reasoning for
     * changing them.
     */
    public final double gamma = 0.9;
    public final double learningRate = .0001;
    public final double epsilon = .02;

    public RLAgent(int playernum, String[] args) {
        super(playernum);

        if (args.length >= 1) {
            numEpisodes = Integer.parseInt(args[0]);
            System.out.println("Running " + numEpisodes + " episodes.");
        } else {
            numEpisodes = 10;
            System.out.println("Warning! Number of episodes not specified. Defaulting to 10 episodes.");
        }

        boolean loadWeights = false;
        if (args.length >= 2) {
            loadWeights = Boolean.parseBoolean(args[1]);
        } else {
            System.out.println("Warning! Load weights argument not specified. Defaulting to not loading.");
        }

        if (loadWeights) {
            weights = loadWeights();
        } else {
            // initialize weights to random values between -1 and 1
            weights = new double[NUM_FEATURES];
            for (int i = 0; i < weights.length; i++) {
                weights[i] = random.nextDouble() * 2 - 1;
            }
        }

        episodesWon = 0;
        currentEpisode = 1;
        episodesTested = 1;
        testingEpisode = true;
        episodesEvaluated = 0;
    }

    /**
     * We've implemented some setup code for your convenience. Change what you need to.
     */
    @Override
    public Map<Integer, Action> initialStep(State.StateView stateView, History.HistoryView historyView) {

        // System.out.printf("episode %4d is a %10s episode\n", currentEpisode, testingEpisode ? "testing" : "evaluation");
        // You will need to add code to check if you are in a testing or learning episode
        if (episodesTested > 9) {
            testingEpisode = false;
            episodesTested = 0;
        } else if (episodesEvaluated > 4) {
            testingEpisode = true;
            episodesEvaluated = 0;
        }

        // we have run all the episodes
        if (currentEpisode > numEpisodes) {
            System.out.printf("\ncurrent: %d total: %d\n", currentEpisode, numEpisodes);
            System.out.printf("Finished running... \nwon %f of games\nexiting\n", ((double) episodesWon / (double) numEpisodes));

            try {
                saveToCSV();
            } catch (Exception e) {
                System.out.println(":(");
            }

            System.exit(0);
        }

        // Find all of your units
        myFootmen = new LinkedList<>();
        for (Integer unitId : stateView.getUnitIds(playernum)) {
            Unit.UnitView unit = stateView.getUnit(unitId);

            String unitName = unit.getTemplateView().getName().toLowerCase();
            if (unitName.equals("footman")) {
                myFootmen.add(unitId);
            } else {
                System.err.println("Unknown unit type: " + unitName);
            }
        }

        // Find all of the enemy units
        enemyFootmen = new LinkedList<>();
        for (Integer unitId : stateView.getUnitIds(ENEMY_PLAYERNUM)) {
            Unit.UnitView unit = stateView.getUnit(unitId);

            String unitName = unit.getTemplateView().getName().toLowerCase();
            if (unitName.equals("footman")) {
                enemyFootmen.add(unitId);
            } else {
                System.err.println("Unknown unit type: " + unitName);
            }
        }

        currentRewards = new ArrayList<>();

        return middleStep(stateView, historyView);
    }

    /**
     * You will need to calculate the reward at each step and update your totals. You will also need to
     * check if an event has occurred. If it has then you will need to update your weights and select a new action.
     *
     * If you are using the footmen vectors you will also need to remove killed units. To do so use the historyView
     * to get a DeathLog. Each DeathLog tells you which player's unit died and the unit ID of the dead unit. To get
     * the deaths from the last turn do something similar to the following snippet. Please be aware that on the first
     * turn you should not call this as you will get nothing back.
     *
     * for(DeathLog deathLog : historyView.getDeathLogs(stateView.getTurnNumber() -1)) {
     *     System.out.println("Player: " + deathLog.getController() + " unit: " + deathLog.getDeadUnitID());
     * }
     *
     * You should also check for completed actions using the history view. Obviously you never want a footman just
     * sitting around doing nothing (the enemy certainly isn't going to stop attacking). So at the minimum you will
     * have an even whenever one your footmen's targets is killed or an action fails. Actions may fail if the target
     * is surrounded or the unit cannot find a path to the unit. To get the action results from the previous turn
     * you can do something similar to the following. Please be aware that on the first turn you should not call this
     *
     * Map<Integer, ActionResult> actionResults = historyView.getCommandFeedback(playernum, stateView.getTurnNumber() - 1);
     * for(ActionResult result : actionResults.values()) {
     *     System.out.println(result.toString());
     * }
     *
     * @return New actions to execute or nothing if an event has not occurred.
     */
    @Override
    public Map<Integer, Action> middleStep(State.StateView stateView, History.HistoryView historyView) {

        // remove dead footmen
        updateFootmenList(myFootmen, stateView, historyView);
        updateFootmenList(enemyFootmen, stateView, historyView);
        updateActions(stateView, historyView);
        Map<Integer, Action> actions = new HashMap<>();

        if (eventOccured(stateView, historyView)) {

            // calculate the rewards
            double reward = 0;
            for (int footmanID : myFootmen) {
                reward += calculateReward(stateView, historyView, footmanID);
            }
            currentRewards.add(reward);

            // if we are in a testing episode then update the policy
            if (testingEpisode) {
                for (int footmanID : myFootmen) {
                    // TODO: NOT SURE IF I SHOULD BE CALCULATING THE FEATURE VECTOR HERE
                    // WE NEED THE OLD FEATURES
                    double[] featureValues = convertToPrimitiveArray(previousFeatureValues.get(footmanID));
                    weights = updateWeights(weights, featureValues, reward, stateView, historyView, footmanID);
                }
            }

            // All footmen get a new action
            for (int footmanID : myFootmen) {
                Action action = Action.createCompoundAttack(footmanID, selectAction(stateView, historyView, footmanID));
                actions.put(footmanID, action);
            }

        } else if (stateView.getTurnNumber() == 0) {
            // First turn give everyone an action

            for (int footmanID : myFootmen) {
                Action action = Action.createCompoundAttack(footmanID, selectAction(stateView, historyView, footmanID));
                actions.put(footmanID, action);
            }

        } else {
            // No event occured so find lazy footmen and put them to work
            Map<Integer, ActionResult> actionResults = historyView.getCommandFeedback(playernum, stateView.getTurnNumber() - 1);

            for(ActionResult result : actionResults.values()) {
                int unitID = result.getAction().getUnitId();
                if (result.getFeedback().equals(ActionFeedback.COMPLETED) && myFootmen.contains(unitID)) {
                    Action action = Action.createCompoundAttack(unitID, selectAction(stateView, historyView, unitID));
                    actions.put(unitID, action);
                }
            }
        }
        return actions;
    }

    /**
     * Here you will calculate the cumulative average rewards for your testing episodes. If you have just
     * finished a set of test episodes you will call out testEpisode.
     *
     * It is also a good idea to save your weights with the saveWeights function.
     */
    @Override
    public void terminalStep(State.StateView stateView, History.HistoryView historyView) {

        currentEpisode++;
        double sumRewards = sum(currentRewards);
        allRewards.add(sumRewards);

        if (testingEpisode) {
            episodesTested++;
        } else {
            episodesEvaluated++;

            // if we are evaluating then add the average to evaluation rewards
            evaluationRewards.add(sumRewards);

            if (episodesEvaluated > 4) {
                averageRewards.add(average(evaluationRewards));
                evaluationRewards = new ArrayList<>();
                printTestData(averageRewards);
                for (int i = 0; i < weights.length; i++) {
                    System.out.printf("%-17s: %f\n", featureNames[i], weights[i]);
                }
                if (averageRewards.get(averageRewards.size() - 1) > bestReward) {
                    bestReward = averageRewards.get(averageRewards.size() - 1);
                    bestWeights = weights;
                    saveBestWeights(bestWeights);

                }
            }
        }

        //System.out.printf("episode %4d %4s\n", currentEpisode, myFootmen.size() > enemyFootmen.size() ? "won" : "lost");
        //System.out.printf("\t%d events occured\n", currentRewards.size());

        if (stateView.getUnits(0).size() > stateView.getUnits(1).size()) {
            episodesWon++;
        }

        // Save your weights
        saveWeights(weights);

    }

    /**
     * Calculate the updated weights for this agent. 
     * @param oldWeights Weights prior to update
     * @param oldFeatures Features from (s,a)
     * @param totalReward Cumulative discounted reward for this footman.
     * @param stateView Current state of the game.
     * @param historyView History of the game up until this point
     * @param footmanId The footman we are updating the weights for
     * @return The updated weight vector.
     */
    public double[] updateWeights(double[] oldWeights,
                                  double[] oldFeatures,
                                  double totalReward,
                                  State.StateView stateView,
                                  History.HistoryView historyView,
                                  int footmanId) {

        double[] newWeights = new double[oldWeights.length];

        for (int i = 0; i < newWeights.length; i++) {
            double oldQValue =  dotProduct(oldWeights, oldFeatures);
            double currentQValue = findMaxQValue(stateView, historyView, footmanId);

            newWeights[i] = oldWeights[i] + learningRate * (totalReward + (gamma * currentQValue) - oldQValue) * oldFeatures[i];
        }

        // System.out.printf("old: %s\nnew: %s\n\n", Arrays.toString(oldWeights), Arrays.toString(newWeights));
        return newWeights;
    }

    /**
     * Calculates the dot product of weights and features.
     * @param weights the weight vector
     * @param features the feature vector
     * @return the dot product
     */
    private double dotProduct(double[] weights, double[] features) {
        double product = 0;

        //System.out.println("dotProduct:");
        //System.out.printf("\t%d weights\n\t%d features", weights.length, features.length);
        for (int i = 0; i < weights.length; i++) {
            product += (weights[i] * features[i]);
        }

        return product;
    }

    /**
     * Finds the maximum Q value for all enemy ids
     * @param stateView Current state of the game.
     * @param historyView History of the game up until this point
     * @param footmanId The footman we are updating the weights for
     * @return the maximum Q value.
     */
    private double findMaxQValue(State.StateView stateView, History.HistoryView historyView, int footmanId) {
        double maxQValue = 0;

        for (int enemyId : enemyFootmen) {
            double newQValue = calcQValue(stateView, historyView, footmanId, enemyId);
            if (newQValue > maxQValue) {
                maxQValue = newQValue;
            }
        }

        return maxQValue;
    }

    /**
     * Given a footman and the current state and history of the game select the enemy that this unit should
     * attack. This is where you would do the epsilon-greedy action selection.
     *
     * @param stateView Current state of the game
     * @param historyView The entire history of this episode
     * @param attackerId The footman that will be attacking
     * @return The enemy footman ID this unit should attack
     */
    public int selectAction(State.StateView stateView, History.HistoryView historyView, int attackerId) {

        Unit.UnitView attacker = stateView.getUnit(attackerId);
        int victim  = enemyFootmen.get(0);

        if (random.nextDouble() < epsilon) {
            // do random stuff
            int victimIndex = (int)(Math.random() * enemyFootmen.size());
            victim = enemyFootmen.get(victimIndex);
        } else {

            double maxQVal = Integer.MIN_VALUE;

            // loop through all of the enemy footmen and figure out which one to attack
            for (int enemyID : enemyFootmen) {

                double newQVal = calcQValue(stateView, historyView, attackerId, enemyID);
                if (newQVal > maxQVal) {
                    victim  = enemyID;
                    maxQVal = newQVal;
                }
            }
        }

        double[] featureVector = calculateFeatureVector(stateView, historyView, attackerId,  victim);
        //System.out.println(Arrays.toString(featureVector));
        previousFeatureValues.put(attackerId, convertToObjectArray(featureVector));
        return victim;
    }

    /**
     * Given the current state and the footman in question calculate the reward received on the last turn.
     * This is where you will check for things like Did this footman take or give damage? Did this footman die
     * or kill its enemy. Did this footman start an action on the last turn? See the assignment description
     * for the full list of rewards.
     *
     * Remember that you will need to discount this reward based on the timestep it is received on. See
     * the assignment description for more details.
     *
     * As part of the reward you will need to calculate if any of the units have taken damage. You can use
     * the history view to get a list of damages dealt in the previous turn. Use something like the following.
     *
     * for(DamageLog damageLogs : historyView.getDamageLogs(lastTurnNumber)) {
     *     System.out.println("Defending player: " + damageLog.getDefenderController() + " defending unit: " + \
     *     damageLog.getDefenderID() + " attacking player: " + damageLog.getAttackerController() + \
     *     "attacking unit: " + damageLog.getAttackerID());
     * }
     *
     * You will do something similar for the deaths. See the middle step documentation for a snippet
     * showing how to use the deathLogs.
     *
     * To see if a command was issued you can check the commands issued log.
     *
     * Map<Integer, Action> commandsIssued = historyView.getCommandsIssued(playernum, lastTurnNumber);
     * for (Map.Entry<Integer, Action> commandEntry : commandsIssued.entrySet()) {
     *     System.out.println("Unit " + commandEntry.getKey() + " was command to " + commandEntry.getValue().toString);
     * }
     *
     * @param stateView The current state of the game.
     * @param historyView History of the episode up until this turn.
     * @param footmanId The footman ID you are looking for the reward from.
     * @return The current reward
     */
    public double calculateReward(State.StateView stateView, History.HistoryView historyView, int footmanId) {
        double reward = -0.1;

        for (DamageLog damageLogs : historyView.getDamageLogs(stateView.getTurnNumber() - 1)) {
            // Footman is being attacked
            if (footmanId == damageLogs.getDefenderID()) {
                if (footmanDied(stateView, historyView, footmanId)) {
                    reward -= 100;
                }

                reward -= damageLogs.getDamage();
            }

            // Footman is attacking
            if (footmanId == damageLogs.getAttackerID()) {
                if (footmanDied(stateView, historyView, damageLogs.getDefenderID())) {
                    reward += 100;
                }

                reward += damageLogs.getDamage();
            }
        }

        return reward;
    }

    /**
     * Returns whether the given footman died.
     * @param stateView The current state of the game.
     * @param historyView History of the episode up until this turn.
     * @param footmanId The footman ID you are looking for the reward from.
     * @return true if the footman died
     */
    private boolean footmanDied(State.StateView stateView, History.HistoryView historyView, int footmanId) {
        for(DeathLog deathLog : historyView.getDeathLogs(stateView.getTurnNumber() - 1)) {
            if (footmanId == deathLog.getDeadUnitID()) {
                return true;
            }
        }
        return false;
    }

    /**
     * Calculate the Q-Value for a given state action pair. The state in this scenario is the current
     * state view and the history of this episode. The action is the attacker and the enemy pair for the
     * SEPIA attack action.
     *
     * This returns the Q-value according to your feature approximation. This is where you will calculate
     * your features and multiply them by your current weights to get the approximate Q-value.
     *
     * @param stateView Current SEPIA state
     * @param historyView Episode history up to this point in the game
     * @param attackerId Your footman. The one doing the attacking.
     * @param defenderId An enemy footman that your footman would be attacking
     * @return The approximate Q-value
     */
    public double calcQValue(State.StateView stateView,
                             History.HistoryView historyView,
                             int attackerId,
                             int defenderId) {
        double [] features = calculateFeatureVector(stateView, historyView, attackerId, defenderId);
        return dotProduct(weights, features);
    }

    /**
     * Given a state and action calculate your features here. Please include a comment explaining what features
     * you chose and why you chose them.
     *
     * All of your feature functions should evaluate to a double. Collect all of these into an array. You will
     * take a dot product of this array with the weights array to get a Q-value for a given state action.
     *
     * It is a good idea to make the first value in your array a constant. This just helps remove any offset
     * from 0 in the Q-function. The other features are up to you. Many are suggested in the assignment
     * description.
     *
     * @param stateView   Current state of the SEPIA game
     * @param historyView History of the game up until this turn
     * @param attackerId  Your footman. The one doing the attacking.
     * @param defenderId  An enemy footman. The one you are considering attacking.
     * @return            The array of feature function outputs.
     */
    public double[] calculateFeatureVector(State.StateView stateView,
                                           History.HistoryView historyView,
                                           int attackerId,
                                           int defenderId) {

        double[] featureVector = new double[8];
        featureVector[0] = .1;

        for (Action action : currentActions) {
            if (action instanceof TargetedAction) {
                TargetedAction targeted = (TargetedAction) action;

                // how many other footmen are attacking e?
                if (targeted.getTargetId() == defenderId) {
                    featureVector[NUM_ATTACKING_FOOTMEN_FEATURE]++;
                }

                // is e attacking me?
                if (targeted.getUnitId() == defenderId && targeted.getTargetId() == attackerId) {
                    featureVector[BEING_ATTACKED_FEATURE]++;
                }

                // is e the closest enemy?
                if (defenderId == getClosestEnemy(attackerId, stateView,historyView)) {
                    featureVector[CLOSEST_ENEMY_FEATURE]++;
                }

            }
        }

        // how much health do i have?
        featureVector[HEALTH_FEATURE] = (stateView.getUnit(attackerId).getHP() - stateView.getUnit(defenderId).getHP());

        // weakest one
        if (defenderId == getWeakestEnemy(stateView)) {
            featureVector[WEAKEST_ENEMY_FEATURE] += 1;
        }

        return featureVector;
    }

    /**
     * Removes any footmen based on death.
     * @param footmen     the footmen list being updated
     * @param stateView   the current state
     * @param historyView the history
     */
    private void updateFootmenList(List<Integer> footmen, State.StateView stateView, History.HistoryView historyView) {

        for(DeathLog deathLog : historyView.getDeathLogs(stateView.getTurnNumber() - 1)) {
            if (footmen.contains(deathLog.getDeadUnitID())) {
                for (int i = 0; i < footmen.size(); i++) {
                    if (footmen.get(i) == deathLog.getDeadUnitID()) {
                        footmen.remove(i);
                    }
                }
            }
        }
    }

    /**
     * Converts double array to Double array
     * @param array the double array
     * @return the Double array
     */
    private Double[] convertToObjectArray(double[] array) {
        Double[] newArray = new Double[array.length];
        for (int i = 0; i < array.length; i++) {
            newArray[i] = array[i];
        }
        return newArray;
    }

    /**
     * Converts Double array to double array
     * @param array the Double array
     * @return the double array
     */
    private double[] convertToPrimitiveArray(Double[] array) {
        double[] newArray = new double[array.length];
        for (int i = 0; i < array.length; i++) {
            newArray[i] = array[i];
        }
        return newArray;
    }

    /**
     * DO NOT CHANGE THIS!
     *
     * Prints the learning rate data described in the assignment. Do not modify this method.
     *
     * @param averageRewards List of cumulative average rewards from test episodes.
     */
    public void printTestData (List<Double> averageRewards) {
        System.out.println("");
        System.out.println("Games Played      Average Cumulative Reward");
        System.out.println("-------------     -------------------------");
        for (int i = 0; i < averageRewards.size(); i++) {
            String gamesPlayed = Integer.toString(10*i);
            String averageReward = String.format("%.2f", averageRewards.get(i));

            int numSpaces = "-------------     ".length() - gamesPlayed.length();
            StringBuffer spaceBuffer = new StringBuffer(numSpaces);
            for (int j = 0; j < numSpaces; j++) {
                spaceBuffer.append(" ");
            }
            System.out.println(gamesPlayed + spaceBuffer.toString() + averageReward);
        }
        System.out.println("");
    }

    /**
     * DO NOT CHANGE THIS!
     *
     * This function will take your set of weights and save them to a file. Overwriting whatever file is
     * currently there. You will use this when training your agents. You will include th output of this function
     * from your trained agent with your submission.
     *
     * Look in the agent_weights folder for the output.
     *
     * @param weights Array of weights
     */
    public void saveWeights(double[] weights) {
        File path = new File("agent_weights/weights.txt");
        // create the directories if they do not already exist
        path.getAbsoluteFile().getParentFile().mkdirs();

        try {
            // open a new file writer. Set append to false
            BufferedWriter writer = new BufferedWriter(new FileWriter(path, false));

            for (double weight : weights) {
                writer.write(String.format("%f\n", weight));
            }
            writer.flush();
            writer.close();
        } catch(IOException ex) {
            System.err.println("Failed to write weights to file. Reason: " + ex.getMessage());
        }
    }

    /**
     * DO NOT CHANGE THIS!
     *
     * This function will load the weights stored at agent_weights/weights.txt. The contents of this file
     * can be created using the saveWeights function. You will use this function if the load weights argument
     * of the agent is set to 1.
     *
     * @return The array of weights
     */
    public double[] loadWeights() {
        File path = new File("agent_weights/weights.txt");
        if (!path.exists()) {
            System.err.println("Failed to load weights. File does not exist");
            return null;
        }

        try {
            BufferedReader reader = new BufferedReader(new FileReader(path));
            String line;
            List<Double> weights = new LinkedList<>();
            while((line = reader.readLine()) != null) {
                weights.add(Double.parseDouble(line));
            }
            reader.close();

            double[] newWeights = new double[weights.size()];
            for (int i = 0; i < weights.size(); i++) {
                newWeights[i] = weights.get(i);
            }
            return newWeights;
        } catch(IOException ex) {
            System.err.println("Failed to load weights from file. Reason: " + ex.getMessage());
        }
        return null;
    }

    public void saveBestWeights(double[] weights) {
        File path = new File("agent_weights/bestweights.data");
        // create the directories if they do not already exist
        path.getAbsoluteFile().getParentFile().mkdirs();

        try {
            // open a new file writer. Set append to false
            BufferedWriter writer = new BufferedWriter(new FileWriter(path, false));

            for (double weight : weights) {
                writer.write(String.format("%f\n", weight));
            }
            writer.flush();
            writer.close();
        } catch(IOException ex) {
            System.err.println("Failed to write weights to file. Reason: " + ex.getMessage());
        }
    }

    /**
     * returns the chebyshev distance between two units
     *
     * @param first  The first unit
     * @param second The second unit
     * @return       The chebyshev distance between the first and second
     */
    public int chebyshev(UnitView first, UnitView second) {

        int deltaY = Math.abs(first.getYPosition() - second.getYPosition());
        int deltaX = Math.abs(first.getXPosition() - second.getXPosition());

        return deltaX > deltaY ? deltaX : deltaY;
    }

    /**
     * Returns if an event has occured
     * @param stateView   the current state
     * @param historyView the history
     * @return            if an event has occured
     */
    public boolean eventOccured(State.StateView stateView, History.HistoryView historyView) {

        // unit is killed
        if (historyView.getDeathLogs(stateView.getTurnNumber() - 1).size() > 0) {
            return true;
        }

        // friendly unit is hit
        for (DamageLog damage : historyView.getDamageLogs(stateView.getTurnNumber() - 1)) {
            if (myFootmen.contains(damage.getDefenderID())) {
                return true;
            }
        }

        return false;
    }

    /**
     * updates the list of actions we are maintaining
     * @param stateView   the current state
     * @param historyView the history
     */
    public void updateActions(State.StateView stateView, History.HistoryView historyView) {

        currentActions = new ArrayList<>();

        Map<Integer, ActionResult> actionResults = historyView.getCommandFeedback(playernum, stateView.getTurnNumber() - 1);
        for (ActionResult result : actionResults.values()) {
            if (!result.equals(ActionFeedback.COMPLETED)) {
                currentActions.add(result.getAction());
            }
        }
    }

    /**
     * returns the closest enemyID to the given footman
     * @param footman     the footman we are looking from
     * @param stateView   the current state
     * @param historyView the history
     * @return the ID of the closest enemy
     */
    private int getClosestEnemy(int footman, State.StateView stateView, History.HistoryView historyView) {

        int closestEnemy = enemyFootmen.get(0);
        int closestDistance = 10000;
        int enemyDistance;
        for (int enemy : enemyFootmen) {
            enemyDistance = chebyshev(stateView.getUnit(footman), stateView.getUnit(enemy));
            if (enemyDistance < closestDistance) {
                closestDistance = enemyDistance;
                closestEnemy = enemy;
            }
        }
        return closestEnemy;
    }

    /**
     * returns the id of the weakest enemy
     * @param stateView the state
     * @return the id of the weakest enemy
     */
    private int getWeakestEnemy(State.StateView stateView) {

        int weakestHealth = Integer.MAX_VALUE;
        int weakEnemy = enemyFootmen.get(0);

        for (int enemyID : enemyFootmen) {
            int enemyHealth = stateView.getUnit(enemyID).getHP();
            if (enemyHealth < weakestHealth) {
                weakestHealth = enemyHealth;
                weakEnemy = enemyID;
            }
        }

        return weakEnemy;
    }

    /**
     * Calculates the average of a list of doubles
     * @param numbers the list of doubles
     * @return the average
     */
    private double average(List<Double> numbers) {
        double average = 0;

        for (double num : numbers) {
            average += num;
        }
        return average / numbers.size();
    }

    /**
     * sums a list of doubles
     * @param numbers the numbers to sum
     * @return the sum of the numbers
     */
    private double sum(List<Double> numbers) {
        double sum = 0;

        for (double num : numbers) {
            sum += num;
        }

        return sum;
    }

    private void saveToCSV() throws IOException {
        FileWriter fileWriter = new FileWriter("results.csv");
        fileWriter.write("iteration, cumulative reward\n");

        int i = 0;
        for (double reward : allRewards) {
            i++;
            fileWriter.write(String.format("%d, %.5f\n", i, reward));
        }

        fileWriter.flush();
        fileWriter.close();
    }

    @Override
    public void savePlayerData(OutputStream outputStream) {

    }

    @Override
    public void loadPlayerData(InputStream inputStream) {

    }
}
