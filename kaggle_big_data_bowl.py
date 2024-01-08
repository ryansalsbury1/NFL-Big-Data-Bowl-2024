#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 22:55:04 2024

@author: ryansalsbury
"""

import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

games = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2024/games.csv")
tracking1 = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2024/tracking_week_1.csv")
tracking2 = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2024/tracking_week_2.csv")
tracking3 = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2024/tracking_week_3.csv")
tracking4 = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2024/tracking_week_4.csv")
tracking5 = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2024/tracking_week_5.csv")
tracking6 = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2024/tracking_week_6.csv")
tracking7 = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2024/tracking_week_7.csv")
tracking8 = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2024/tracking_week_8.csv")
tracking9 = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2024/tracking_week_9.csv")
plays = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2024/plays.csv")
players = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2024/players.csv")
#make full play description visible
pd.set_option('display.max_colwidth', None)

#combine all tracking data
tracking = pd.concat([tracking1, tracking2, tracking3, tracking4, tracking5, tracking6, tracking7, tracking8, tracking9])

#make full play description visible
pd.set_option('display.max_colwidth', None)
#display all columns
pd.set_option('display.max_columns', None)
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

#get play information for tracking data
tracking = pd.merge(plays[['gameId', 'playId', 'playDescription', 'possessionTeam', 'ballCarrierId', 'ballCarrierDisplayName', 'passResult', 'playNullifiedByPenalty', 'yardlineSide', 'yardlineNumber', 'yardsToGo']], tracking, on = ['gameId', 'playId'])

#convert yard line number to same scale as x coordinates (in order to calculate yards to go after catch is made)
tracking['yardlineNumber'] = np.where(tracking['yardlineSide'] == tracking['possessionTeam'], tracking['yardlineNumber'] + 10, 60 + (50-tracking['yardlineNumber']))

#filter to only plays where there was a completed pass that did not get nullified by a penalty
completed_pass_plays = tracking[(tracking['passResult'] == 'C') & (tracking['playNullifiedByPenalty'] == 'N')].copy()

#add binary pass_catcher column for whether or not the row represents the ball carrier of the play
completed_pass_plays['pass_catcher'] = np.where(completed_pass_plays['nflId'] == completed_pass_plays['ballCarrierId'], 1,0)

#add on_offense binary column for whether or not the row represents an offensive player
completed_pass_plays['on_offense'] = np.where(completed_pass_plays['possessionTeam'] == completed_pass_plays['club'], 1,0)

#rename team and player name columns
completed_pass_plays.rename(columns={'club':'team', 'displayName': 'player', 'possessionTeam': 'offense_team'}, inplace=True)

#standardize plays so that offense is always moving left to right
#x and y
completed_pass_plays['x'] = np.where(completed_pass_plays['playDirection'] == 'left', 120 - completed_pass_plays['x'],completed_pass_plays['x'])
completed_pass_plays['y'] = np.where(completed_pass_plays['playDirection'] == 'left', 53.3 - completed_pass_plays['y'],completed_pass_plays['y'])

#orientation (0 degrees is facing up, 90 degrees is right, 180 degrees is down, 270 degrees is left; negative degrees for 270 to 0)
conditions_o = [(completed_pass_plays['playDirection'] == 'left') & (completed_pass_plays['o'] < 90), (completed_pass_plays['playDirection'] == 'left') & (completed_pass_plays['o'] > 90), (completed_pass_plays['playDirection'] == 'right') & (completed_pass_plays['o'] > 270)]
values_o = [(360 + completed_pass_plays['o']) - 180, completed_pass_plays['o'] - 180, completed_pass_plays['o'] - 360]
completed_pass_plays['o'] = np.select(conditions_o, values_o, default=completed_pass_plays['o'])

#dir (0 degrees is facing up, 90 degrees is right, 180 degrees is down, 270 degrees is left; negative degrees for 270 to 0)
conditions_dir = [(completed_pass_plays['playDirection'] == 'left') & (completed_pass_plays['dir'] < 90), (completed_pass_plays['playDirection'] == 'left') & (completed_pass_plays['dir'] > 90), (completed_pass_plays['playDirection'] == 'right') & (completed_pass_plays['dir'] > 270)]
values_dir = [(360 + completed_pass_plays['dir']) - 180, completed_pass_plays['dir'] - 180, completed_pass_plays['dir'] - 360]
completed_pass_plays['dir'] = np.select(conditions_dir, values_dir, default=completed_pass_plays['dir'])


#get all frames where ball was caught and the outcome of the play (fumbles have either a tackle event associated with it or a fumble defense recovered event associated with it - For this exercise, just looking at either a tackle event or fumble defense recovered event if a fumble occured.)
#don't include frameId for Greg Dortch reception where there is duplicate row (gameId: 2022110608 playId: 2351)
caught_pass_events = completed_pass_plays[(completed_pass_plays['event'] == 'pass_outcome_caught') & (completed_pass_plays['frameId'] != 32)]

#get frames where pass arrived for pass catcher (used to try to assess accuracy of pass)
pass_arrival_events = completed_pass_plays[((completed_pass_plays['event'] == 'pass_arrived') | (completed_pass_plays['frameId'] == 4))  & (completed_pass_plays['pass_catcher'] == 1)][['gameId', 'playId', 's', 'o', 'dir', 'event']]
pass_arrival_events = pass_arrival_events.sort_values('event', ascending=False).groupby(['gameId', 'playId'], as_index=False).first()[['gameId', 'playId', 's', 'o', 'dir']]
pass_arrival_events.rename(columns={'s': 's_pass_catcher_arrival', 'o': 'o_pass_catcher_arrival', 'dir': 'dir_pass_catcher_arrival'}, inplace=True)


#get outcome of all plays
#create outcome table for ball carriers where event is not null
play_outcomes = completed_pass_plays[(completed_pass_plays['pass_catcher'] == 1) & (completed_pass_plays['event'].isna() == False)]

#get all unique events for a given play
all_events = play_outcomes[['gameId', 'playId', 'event']].groupby(['gameId', 'playId'], as_index=False).agg(', '.join).rename(columns={'event':'all_events'})

#add all_events column to play_outcomes table
play_outcomes = pd.merge(play_outcomes, all_events, on=['gameId', 'playId'])
play_outcomes = play_outcomes[((play_outcomes['all_events'].str.contains('fumble')) & (play_outcomes['event'] == 'fumble')) | (~(play_outcomes['all_events'].str.contains('fumble')) & (play_outcomes['event'].isin(['tackle', 'out_of_bounds', 'touchdown', 'pass_outcome_touchdown'])))][['gameId', 'playId', 'x', 'y', 'event']]
play_outcomes.rename(columns={'x': 'x_outcome', 'y': 'y_outcome', 'event': 'event_outcome'}, inplace=True)

#get player position
caught_pass_events = pd.merge(caught_pass_events, players[['nflId', 'position']], on='nflId')

#group position into more general positions
conditions_pos = [caught_pass_events['position'].isin(['T','C','G']),  caught_pass_events['position'].isin(['DT','NT','DE']),  caught_pass_events['position'].isin(['OLB','MLB','ILB']),  caught_pass_events['position'].isin(['CB','SS','FS','DB'])]
values_pos = ['OL', 'DL', 'LB', 'DB']
caught_pass_events['position'] = np.select(conditions_pos, values_pos, default=caught_pass_events['position'])

#create pass catchers df for only players with the ball (used to merge to other table and calculate distance)
pass_catchers = caught_pass_events[caught_pass_events['pass_catcher'] == 1]
#add suffix to columns to represent pass catcher
pass_catchers = pass_catchers.rename(columns={c: c+'_pass_catcher' for c in pass_catchers.columns if c in ['player', 'x', 'y', 's', 'a', 'dis', 'o', 'dir', 'position']})

#merge the plays to pass_catchers to join on pass catcher with all other players in order to calculate distance
caught_pass_events = pd.merge(caught_pass_events, pass_catchers[['gameId', 'playId', 'player_pass_catcher', 'x_pass_catcher', 'y_pass_catcher', 's_pass_catcher', 'a_pass_catcher',
       'dis_pass_catcher', 'o_pass_catcher', 'dir_pass_catcher', 'position_pass_catcher']], on=['gameId', 'playId'])

#remove 2 rows for each play with ball carrier and football
caught_pass_events = caught_pass_events[(caught_pass_events['pass_catcher'] == 0) & (caught_pass_events['player'] != 'football')]

#calculate distance in feet between each player and ball carrier + calculate distance from reciever to nearest sideline + calculate distance from reciever to opponent endzone
import math
def distance(x1 , y1 , x2 , y2): 
    # Calculating distance 
    return math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2) * 1.0) 


caught_pass_events['distance'] = caught_pass_events.apply(lambda x: distance(x.x, x.y, x.x_pass_catcher, x.y_pass_catcher), axis=1)
caught_pass_events['catch_distance_from_sideline'] = np.where(caught_pass_events['y_pass_catcher'] >= 26.65, 53.3-caught_pass_events['y_pass_catcher'], caught_pass_events['y_pass_catcher'])
caught_pass_events['catch_distance_from_endzone'] = 110-caught_pass_events['x_pass_catcher']
caught_pass_events['catch_distance_from_los'] = caught_pass_events['x_pass_catcher'] - caught_pass_events['yardlineNumber']
#yards left to go for first down at point of catch
caught_pass_events['yardsToGo'] = caught_pass_events['yardsToGo'] - caught_pass_events['catch_distance_from_los']


#add whether or not player is behind or ahead (closer to defending endzone) of pass catcher
caught_pass_events['ahead_of_pass_catcher'] = np.where(caught_pass_events['x'] >= caught_pass_events['x_pass_catcher'], 1, 0)
off_players_ahead_of_pass_catcher = caught_pass_events[caught_pass_events['on_offense'].eq(1)].groupby(['gameId', 'playId'], as_index=False)['ahead_of_pass_catcher'].sum().rename(columns={'ahead_of_pass_catcher': 'off_players_ahead_of_pass_catcher'})
def_players_ahead_of_pass_catcher = caught_pass_events[caught_pass_events['on_offense'].eq(0)].groupby(['gameId', 'playId'], as_index=False)['ahead_of_pass_catcher'].sum().rename(columns={'ahead_of_pass_catcher': 'def_players_ahead_of_pass_catcher'})
caught_pass_events = pd.merge(caught_pass_events, off_players_ahead_of_pass_catcher)
caught_pass_events = pd.merge(caught_pass_events, def_players_ahead_of_pass_catcher)

#get distance ranks for each offensive and defensive player (priority given to offensive players that are ahead of pass catcher and in a position to block)
caught_pass_events['dist_rank_off'] = caught_pass_events[caught_pass_events['on_offense'].eq(1)].sort_values(by=['ahead_of_pass_catcher', 'distance', 'gameId', 'playId', 'team'], ascending=[False, True, True, True, True]).groupby(['gameId', 'playId', 'team'])['distance'].cumcount(ascending=True) + 1
caught_pass_events['dist_rank_def'] = caught_pass_events[caught_pass_events['on_offense'].eq(0)].groupby(['gameId', 'playId', 'team'])['distance'].rank(method="dense", ascending=True).astype(int)
caught_pass_events['dist_rank'] = caught_pass_events.dist_rank_off.combine_first(caught_pass_events.dist_rank_def).astype(int)

#keep only columns needed
caught_pass_events  = caught_pass_events[['gameId', 'playId', 'playDescription', 'offense_team', 'team', 'on_offense', 'player', 'position', 'x', 'y',
       's', 'a', 'dis', 'o', 'dir', 'player_pass_catcher', 'position_pass_catcher', 'x_pass_catcher',
       'y_pass_catcher', 's_pass_catcher', 'a_pass_catcher',
       'dis_pass_catcher', 'o_pass_catcher', 'dir_pass_catcher', 'ahead_of_pass_catcher', 'distance', 'catch_distance_from_sideline', 'catch_distance_from_endzone', 'catch_distance_from_los',  'yardsToGo', 'off_players_ahead_of_pass_catcher', 'def_players_ahead_of_pass_catcher',
       'dist_rank']]

#split by offense and defense and rename columns
caught_pass_events_offense = caught_pass_events[caught_pass_events['on_offense'] == 1].drop(['on_offense', 'team'], axis=1)
caught_pass_events_defense = caught_pass_events[caught_pass_events['on_offense'] == 0].drop(['on_offense'], axis=1).rename(columns={'team': 'defense_team'})


#drop on_offense column from caught_pass_events table
caught_pass_events.drop(['on_offense'], axis=1, inplace=True)

#add offense suffix
caught_pass_events_offense = caught_pass_events_offense.rename(columns={c: c+'_off' for c in caught_pass_events_offense.columns if c in ['player', 'position', 'x', 'y',
       's', 'a', 'dis', 'o', 'dir', 'ahead_of_pass_catcher', 'distance',
       'dist_rank']})

#add defense suffix
caught_pass_events_defense = caught_pass_events_defense.rename(columns={c: c+'_def' for c in caught_pass_events_defense.columns if c in ['player', 'position', 'x', 'y',
       's', 'a', 'dis', 'o', 'dir', 'ahead_of_pass_catcher', 'distance',
       'dist_rank']})


#pivot data to one row per play per game
caught_pass_events_off_pivoted = (caught_pass_events_offense.sort_values(['gameId', 'playId', 'dist_rank_off'])
.pivot(index=['gameId', 'playId', 'playDescription', 'offense_team', 'player_pass_catcher', 'position_pass_catcher', 'x_pass_catcher',
       'y_pass_catcher', 's_pass_catcher', 'a_pass_catcher',
       'dis_pass_catcher', 'o_pass_catcher', 'dir_pass_catcher', 'catch_distance_from_sideline', 'catch_distance_from_endzone', 'catch_distance_from_los', 'yardsToGo', 'off_players_ahead_of_pass_catcher', 'def_players_ahead_of_pass_catcher'],columns='dist_rank_off', values=['player_off', 'position_off', 'x_off', 'y_off', 's_off',
       'a_off', 'dis_off', 'o_off', 'dir_off', 'ahead_of_pass_catcher_off', 'distance_off', 'dist_rank_off'])
.sort_index(axis=1, level=1))
caught_pass_events_off_pivoted.columns = [f'{a}_{b}' for a, b in caught_pass_events_off_pivoted.columns]
caught_pass_events_off_pivoted = caught_pass_events_off_pivoted.reset_index()


caught_pass_events_def_pivoted = (caught_pass_events_defense.sort_values(['gameId', 'playId', 'dist_rank_def'])
.pivot(index=['gameId', 'playId', 'playDescription', 'defense_team', 'player_pass_catcher', 'position_pass_catcher', 'x_pass_catcher',
       'y_pass_catcher', 's_pass_catcher', 'a_pass_catcher',
       'dis_pass_catcher', 'o_pass_catcher', 'dir_pass_catcher', 'catch_distance_from_sideline', 'catch_distance_from_endzone', 'catch_distance_from_los', 'yardsToGo', 'off_players_ahead_of_pass_catcher', 'def_players_ahead_of_pass_catcher'],columns='dist_rank_def', values=['player_def', 'position_def', 'x_def', 'y_def', 's_def',
       'a_def', 'dis_def', 'o_def', 'dir_def', 'ahead_of_pass_catcher_def', 'distance_def','dist_rank_def'])
.sort_index(axis=1, level=1))
caught_pass_events_def_pivoted.columns = [f'{a}_{b}' for a, b in caught_pass_events_def_pivoted.columns]
caught_pass_events_def_pivoted = caught_pass_events_def_pivoted.reset_index()

#Combine offense and defense into one table
caught_pass_events_pivoted = pd.merge(caught_pass_events_off_pivoted, caught_pass_events_def_pivoted[['gameId', 'playId', 'defense_team', 'player_def_1', 'a_def_1', 'dir_def_1',
       'dis_def_1', 'ahead_of_pass_catcher_def_1', 'dist_rank_def_1', 'distance_def_1', 'o_def_1',
       'position_def_1', 's_def_1', 'x_def_1', 'y_def_1', 'player_def_2', 'a_def_2',
       'dir_def_2', 'dis_def_2', 'ahead_of_pass_catcher_def_2', 'dist_rank_def_2', 'distance_def_2',
       'o_def_2', 'position_def_2', 's_def_2', 'x_def_2', 'y_def_2', 'player_def_3', 'a_def_3',
       'dir_def_3', 'dis_def_3', 'ahead_of_pass_catcher_def_3', 'dist_rank_def_3', 'distance_def_3',
       'o_def_3', 'position_def_3', 's_def_3', 'x_def_3', 'y_def_3', 'player_def_4', 'a_def_4',
       'dir_def_4', 'dis_def_4', 'ahead_of_pass_catcher_def_4', 'dist_rank_def_4', 'distance_def_4',
       'o_def_4', 'position_def_4', 's_def_4', 'x_def_4', 'y_def_4', 'player_def_5', 'a_def_5',
       'dir_def_5', 'dis_def_5', 'ahead_of_pass_catcher_def_5', 'dist_rank_def_5', 'distance_def_5',
       'o_def_5', 'position_def_5', 's_def_5', 'x_def_5', 'y_def_5', 'player_def_6', 'a_def_6', 'dir_def_6',
       'dis_def_6', 'ahead_of_pass_catcher_def_6', 'dist_rank_def_6', 'distance_def_6', 'o_def_6',
       'position_def_6', 's_def_6', 'x_def_6', 'y_def_6', 'player_def_7', 'a_def_7', 'dir_def_7',
       'dis_def_7', 'ahead_of_pass_catcher_def_7', 'dist_rank_def_7', 'distance_def_7', 'o_def_7',
       'position_def_7', 's_def_7', 'x_def_7', 'y_def_7', 'player_def_8', 'a_def_8', 'dir_def_8',
       'dis_def_8', 'ahead_of_pass_catcher_def_8', 'dist_rank_def_8', 'distance_def_8', 'o_def_8',
       'position_def_8', 's_def_8', 'x_def_8', 'y_def_8', 'player_def_9', 'a_def_9', 'dir_def_9',
       'dis_def_9', 'ahead_of_pass_catcher_def_9', 'dist_rank_def_9', 'distance_def_9', 'o_def_9',
       'position_def_9', 's_def_9', 'x_def_9', 'y_def_9', 'player_def_10', 'a_def_10', 'dir_def_10',
       'dis_def_10', 'ahead_of_pass_catcher_def_10', 'dist_rank_def_10', 'distance_def_10', 'o_def_10',
       'position_def_10', 's_def_10', 'x_def_10', 'y_def_10', 'player_def_11', 'a_def_11', 'dir_def_11',
       'dis_def_11', 'ahead_of_pass_catcher_def_11', 'dist_rank_def_11', 'distance_def_11', 'o_def_11',
       'position_def_11', 's_def_11', 'x_def_11', 'y_def_11',]], on=['gameId', 'playId'])

#add outcome to every play
caught_pass_events_pivoted = pd.merge(caught_pass_events_pivoted, pass_arrival_events, on=['gameId', 'playId'])
caught_pass_events_pivoted = pd.merge(caught_pass_events_pivoted, play_outcomes, on=['gameId', 'playId'])

#calculate change in values from pass arrived to pass caught
caught_pass_events_pivoted['pass_catcher_s_change'] = caught_pass_events_pivoted['s_pass_catcher']- caught_pass_events_pivoted['s_pass_catcher_arrival']
caught_pass_events_pivoted['pass_catcher_o_change'] = caught_pass_events_pivoted['o_pass_catcher'] - caught_pass_events_pivoted['o_pass_catcher_arrival']
caught_pass_events_pivoted['pass_catcher_dir_change'] = caught_pass_events_pivoted['dir_pass_catcher'] - caught_pass_events_pivoted['dir_pass_catcher_arrival']

#calculate distance from caught pass event to end result of play
caught_pass_events_pivoted['outcome_distance'] = caught_pass_events_pivoted['x_outcome'] - caught_pass_events_pivoted['x_pass_catcher']


#reorder columns and keep only columns needed
data = caught_pass_events_pivoted[['gameId', 'playId', 'playDescription', 'offense_team', 'defense_team',
       #pass patcher
       'player_pass_catcher', 'position_pass_catcher',
       'x_pass_catcher', 'y_pass_catcher', 
       's_pass_catcher', 'a_pass_catcher', 'dis_pass_catcher',
       'o_pass_catcher', 'dir_pass_catcher', 'pass_catcher_dir_change', 'pass_catcher_o_change', 'pass_catcher_s_change', 'catch_distance_from_sideline', 
       'catch_distance_from_endzone', 'catch_distance_from_los', 'yardsToGo', 'off_players_ahead_of_pass_catcher', 'def_players_ahead_of_pass_catcher',
       #off player 1
       'player_off_1', 'position_off_1',
       'x_off_1', 'y_off_1', 'ahead_of_pass_catcher_off_1', 's_off_1', 
       'a_off_1', 'dis_off_1', 'o_off_1', 'dir_off_1', 'distance_off_1',
       #off player 2
       'player_off_2', 'position_off_2',
       'x_off_2', 'y_off_2', 'ahead_of_pass_catcher_off_2', 's_off_2',
       'a_off_2', 'dis_off_2', 'o_off_2', 'dir_off_2', 'distance_off_2',
        #off player 3
       'player_off_3', 'position_off_3',
       'x_off_3', 'y_off_3', 'ahead_of_pass_catcher_off_3', 's_off_3', 
       'a_off_3', 'dis_off_3', 'o_off_3', 'dir_off_3', 'distance_off_3',
        #off player 4
       'player_off_4', 'position_off_4',
       'x_off_4', 'y_off_4', 'ahead_of_pass_catcher_off_4', 's_off_4', 
       'a_off_4', 'dis_off_4', 'o_off_4', 'dir_off_4', 'distance_off_4',
       #off player 5
       'player_off_5', 'position_off_5',
       'x_off_5', 'y_off_5', 'ahead_of_pass_catcher_off_5', 's_off_5', 
       'a_off_5', 'dis_off_5', 'o_off_5', 'dir_off_5', 'distance_off_5',
       #off player 6
       'player_off_6', 'position_off_6',
       'x_off_6', 'y_off_6', 'ahead_of_pass_catcher_off_6', 's_off_6', 
       'a_off_6', 'dis_off_6', 'o_off_6', 'dir_off_6', 'distance_off_6',
       #off player 7
       'player_off_7', 'position_off_7',
       'x_off_7', 'y_off_7', 'ahead_of_pass_catcher_off_7', 's_off_7',
       'a_off_7', 'dis_off_7', 'o_off_7', 'dir_off_7', 'distance_off_7',
        #off player 8
       'player_off_8', 'position_off_8',
       'x_off_8', 'y_off_8', 'ahead_of_pass_catcher_off_8', 's_off_8', 
       'a_off_8', 'dis_off_8', 'o_off_8', 'dir_off_8', 'distance_off_8',
        #off player 9
       'player_off_9', 'position_off_9',
       'x_off_9', 'y_off_9', 'ahead_of_pass_catcher_off_9', 's_off_9', 
       'a_off_9', 'dis_off_9', 'o_off_9', 'dir_off_9', 'distance_off_9',
       #off player 10
       'player_off_10', 'position_off_10',
       'x_off_10', 'y_off_10', 'ahead_of_pass_catcher_off_10', 's_off_10', 
       'a_off_10', 'dis_off_10', 'o_off_10', 'dir_off_10', 'distance_off_10',
       #def player 1
       'player_def_1', 'position_def_1',
       'x_def_1', 'y_def_1', 'ahead_of_pass_catcher_def_1', 's_def_1', 
       'a_def_1', 'dis_def_1', 'o_def_1', 'dir_def_1', 'distance_def_1',
       #def player 2
       'player_def_2', 'position_def_2', 
       'x_def_2', 'y_def_2', 'ahead_of_pass_catcher_def_2', 's_def_2', 
       'a_def_2', 'dis_def_2', 'o_def_2', 'dir_def_2', 'distance_def_2',
       #def player 3
       'player_def_3', 'position_def_3',
       'x_def_3', 'y_def_3', 'ahead_of_pass_catcher_def_3', 's_def_3', 
       'a_def_3', 'dis_def_3', 'o_def_3', 'dir_def_3', 'distance_def_3', 
       #def player 4
       'player_def_4', 'position_def_4',
       'x_def_4', 'y_def_4', 'ahead_of_pass_catcher_def_4', 's_def_4', 
       'a_def_4', 'dis_def_4', 'o_def_4', 'dir_def_4', 'distance_def_4', 
       #def player 5
       'player_def_5', 'position_def_5',
       'x_def_5', 'y_def_5', 'ahead_of_pass_catcher_def_5', 's_def_5', 
       'a_def_5', 'dis_def_5', 'o_def_5', 'dir_def_5', 'distance_def_5', 
       #def player 6
       'player_def_6', 'position_def_6',
       'x_def_6', 'y_def_6', 'ahead_of_pass_catcher_def_6', 's_def_6', 
       'a_def_6', 'dis_def_6', 'o_def_6', 'dir_def_6', 'distance_def_6',
       #def player 7
       'player_def_7', 'position_def_7', 
       'x_def_7', 'y_def_7', 'ahead_of_pass_catcher_def_7', 's_def_7', 
       'a_def_7', 'dis_def_7', 'o_def_7', 'dir_def_7', 'distance_def_7',
       #def player 8
       'player_def_8', 'position_def_8',
       'x_def_8', 'y_def_8', 'ahead_of_pass_catcher_def_8', 's_def_8', 
       'a_def_8', 'dis_def_8', 'o_def_8', 'dir_def_8', 'distance_def_8', 
       #def player 9
       'player_def_9', 'position_def_9',
       'x_def_9', 'y_def_9', 'ahead_of_pass_catcher_def_9', 's_def_9', 
       'a_def_9', 'dis_def_9', 'o_def_9', 'dir_def_9', 'distance_def_9', 
       #def player 10
       'player_def_10', 'position_def_10',
       'x_def_10', 'y_def_10', 'ahead_of_pass_catcher_def_10', 's_def_10', 
       'a_def_10', 'dis_def_10', 'o_def_10', 'dir_def_10', 'distance_def_10', 
       #def player 11
       'player_def_11', 'position_def_11',
       'x_def_11', 'y_def_11', 'ahead_of_pass_catcher_def_11', 's_def_11', 
       'a_def_11', 'dis_def_11', 'o_def_11', 'dir_def_11', 'distance_def_11', 
       #outcome
       'event_outcome',
       'outcome_distance']].copy()


#change data types back to numeric
data[['x_pass_catcher', 'y_pass_catcher', 
's_pass_catcher', 'a_pass_catcher', 'dis_pass_catcher',
'o_pass_catcher', 'dir_pass_catcher', 'pass_catcher_dir_change', 'pass_catcher_o_change', 'pass_catcher_s_change', 'catch_distance_from_sideline', 
'catch_distance_from_endzone', 'off_players_ahead_of_pass_catcher', 
'catch_distance_from_los', 'yardsToGo', 'def_players_ahead_of_pass_catcher',
#off player 1
'x_off_1', 'y_off_1', 'ahead_of_pass_catcher_off_1', 's_off_1', 
'a_off_1', 'dis_off_1', 'o_off_1', 'dir_off_1', 'distance_off_1',
#off player 2
'x_off_2', 'y_off_2', 'ahead_of_pass_catcher_off_2', 's_off_2',
'a_off_2', 'dis_off_2', 'o_off_2', 'dir_off_2', 'distance_off_2',
 #off player 3
'x_off_3', 'y_off_3', 'ahead_of_pass_catcher_off_3', 's_off_3', 
'a_off_3', 'dis_off_3', 'o_off_3', 'dir_off_3', 'distance_off_3',
 #off player 4
'x_off_4', 'y_off_4', 'ahead_of_pass_catcher_off_4', 's_off_4', 
'a_off_4', 'dis_off_4', 'o_off_4', 'dir_off_4', 'distance_off_4',
 #off player 5
'x_off_5', 'y_off_5', 'ahead_of_pass_catcher_off_5', 's_off_5', 
'a_off_5', 'dis_off_5', 'o_off_5', 'dir_off_5', 'distance_off_5',
#off player 6
'x_off_6', 'y_off_6', 'ahead_of_pass_catcher_off_6', 's_off_6', 
'a_off_6', 'dis_off_6', 'o_off_6', 'dir_off_6', 'distance_off_6',
#off player 7
'x_off_7', 'y_off_7', 'ahead_of_pass_catcher_off_7', 's_off_7',
'a_off_7', 'dis_off_7', 'o_off_7', 'dir_off_7', 'distance_off_7',
 #off player 8
'x_off_8', 'y_off_8', 'ahead_of_pass_catcher_off_8', 's_off_8', 
'a_off_8', 'dis_off_8', 'o_off_8', 'dir_off_8', 'distance_off_8',
 #off player 9
'x_off_9', 'y_off_9', 'ahead_of_pass_catcher_off_9', 's_off_9', 
'a_off_9', 'dis_off_9', 'o_off_9', 'dir_off_9', 'distance_off_9',
 #off player 10
'x_off_10', 'y_off_10', 'ahead_of_pass_catcher_off_10', 's_off_10', 
'a_off_10', 'dis_off_10', 'o_off_10', 'dir_off_10', 'distance_off_10',
#def player 1
'x_def_1', 'y_def_1', 'ahead_of_pass_catcher_def_1', 's_def_1', 
'a_def_1', 'dis_def_1', 'o_def_1', 'dir_def_1', 'distance_def_1',
#def player 2
'x_def_2', 'y_def_2', 'ahead_of_pass_catcher_def_2', 's_def_2', 
'a_def_2', 'dis_def_2', 'o_def_2', 'dir_def_2', 'distance_def_2',
#def player 3
'x_def_3', 'y_def_3', 'ahead_of_pass_catcher_def_3', 's_def_3', 
'a_def_3', 'dis_def_3', 'o_def_3', 'dir_def_3', 'distance_def_3', 
#def player 4
'x_def_4', 'y_def_4', 'ahead_of_pass_catcher_def_4', 's_def_4', 
'a_def_4', 'dis_def_4', 'o_def_4', 'dir_def_4', 'distance_def_4', 
#def player 5
'x_def_5', 'y_def_5', 'ahead_of_pass_catcher_def_5', 's_def_5', 
'a_def_5', 'dis_def_5', 'o_def_5', 'dir_def_5', 'distance_def_5', 
#def player 6
'x_def_6', 'y_def_6', 'ahead_of_pass_catcher_def_6', 's_def_6', 
'a_def_6', 'dis_def_6', 'o_def_6', 'dir_def_6', 'distance_def_6',
#def player 7
'x_def_7', 'y_def_7', 'ahead_of_pass_catcher_def_7', 's_def_7', 
'a_def_7', 'dis_def_7', 'o_def_7', 'dir_def_7', 'distance_def_7',
#def player 8
'x_def_8', 'y_def_8', 'ahead_of_pass_catcher_def_8', 's_def_8', 
'a_def_8', 'dis_def_8', 'o_def_8', 'dir_def_8', 'distance_def_8', 
#def player 9
'x_def_9', 'y_def_9', 'ahead_of_pass_catcher_def_9', 's_def_9', 
'a_def_9', 'dis_def_9', 'o_def_9', 'dir_def_9', 'distance_def_9', 
#def player 10
'x_def_10', 'y_def_10', 'ahead_of_pass_catcher_def_10', 's_def_10', 
'a_def_10', 'dis_def_10', 'o_def_10', 'dir_def_10', 'distance_def_10', 
#def player 11
'x_def_11', 'y_def_11', 'ahead_of_pass_catcher_def_11', 's_def_11', 
'a_def_11', 'dis_def_11', 'o_def_11', 'dir_def_11', 'distance_def_11', 
#outcome
'outcome_distance']] = data[['x_pass_catcher', 'y_pass_catcher', 
's_pass_catcher', 'a_pass_catcher', 'dis_pass_catcher',
'o_pass_catcher', 'dir_pass_catcher', 'pass_catcher_dir_change', 'pass_catcher_o_change', 'pass_catcher_s_change', 'catch_distance_from_sideline', 
'catch_distance_from_endzone', 'off_players_ahead_of_pass_catcher', 
'catch_distance_from_los', 'yardsToGo', 'def_players_ahead_of_pass_catcher',
#off player 1
'x_off_1', 'y_off_1', 'ahead_of_pass_catcher_off_1', 's_off_1', 
'a_off_1', 'dis_off_1', 'o_off_1', 'dir_off_1', 'distance_off_1',
#off player 2
'x_off_2', 'y_off_2', 'ahead_of_pass_catcher_off_2', 's_off_2',
'a_off_2', 'dis_off_2', 'o_off_2', 'dir_off_2', 'distance_off_2',
 #off player 3
'x_off_3', 'y_off_3', 'ahead_of_pass_catcher_off_3', 's_off_3', 
'a_off_3', 'dis_off_3', 'o_off_3', 'dir_off_3', 'distance_off_3',
 #off player 4
'x_off_4', 'y_off_4', 'ahead_of_pass_catcher_off_4', 's_off_4', 
'a_off_4', 'dis_off_4', 'o_off_4', 'dir_off_4', 'distance_off_4',
 #off player 5
'x_off_5', 'y_off_5', 'ahead_of_pass_catcher_off_5', 's_off_5', 
'a_off_5', 'dis_off_5', 'o_off_5', 'dir_off_5', 'distance_off_5',
#off player 6
'x_off_6', 'y_off_6', 'ahead_of_pass_catcher_off_6', 's_off_6', 
'a_off_6', 'dis_off_6', 'o_off_6', 'dir_off_6', 'distance_off_6',
#off player 7
'x_off_7', 'y_off_7', 'ahead_of_pass_catcher_off_7', 's_off_7',
'a_off_7', 'dis_off_7', 'o_off_7', 'dir_off_7', 'distance_off_7',
 #off player 8
'x_off_8', 'y_off_8', 'ahead_of_pass_catcher_off_8', 's_off_8', 
'a_off_8', 'dis_off_8', 'o_off_8', 'dir_off_8', 'distance_off_8',
 #off player 9
'x_off_9', 'y_off_9', 'ahead_of_pass_catcher_off_9', 's_off_9', 
'a_off_9', 'dis_off_9', 'o_off_9', 'dir_off_9', 'distance_off_9',
 #off player 10
'x_off_10', 'y_off_10', 'ahead_of_pass_catcher_off_10', 's_off_10', 
'a_off_10', 'dis_off_10', 'o_off_10', 'dir_off_10', 'distance_off_10',
#def player 1
'x_def_1', 'y_def_1', 'ahead_of_pass_catcher_def_1', 's_def_1', 
'a_def_1', 'dis_def_1', 'o_def_1', 'dir_def_1', 'distance_def_1',
#def player 2
'x_def_2', 'y_def_2', 'ahead_of_pass_catcher_def_2', 's_def_2', 
'a_def_2', 'dis_def_2', 'o_def_2', 'dir_def_2', 'distance_def_2',
#def player 3
'x_def_3', 'y_def_3', 'ahead_of_pass_catcher_def_3', 's_def_3', 
'a_def_3', 'dis_def_3', 'o_def_3', 'dir_def_3', 'distance_def_3', 
#def player 4
'x_def_4', 'y_def_4', 'ahead_of_pass_catcher_def_4', 's_def_4', 
'a_def_4', 'dis_def_4', 'o_def_4', 'dir_def_4', 'distance_def_4', 
#def player 5
'x_def_5', 'y_def_5', 'ahead_of_pass_catcher_def_5', 's_def_5', 
'a_def_5', 'dis_def_5', 'o_def_5', 'dir_def_5', 'distance_def_5', 
#def player 6
'x_def_6', 'y_def_6', 'ahead_of_pass_catcher_def_6', 's_def_6', 
'a_def_6', 'dis_def_6', 'o_def_6', 'dir_def_6', 'distance_def_6',
#def player 7
'x_def_7', 'y_def_7', 'ahead_of_pass_catcher_def_7', 's_def_7', 
'a_def_7', 'dis_def_7', 'o_def_7', 'dir_def_7', 'distance_def_7',
#def player 8
'x_def_8', 'y_def_8', 'ahead_of_pass_catcher_def_8', 's_def_8', 
'a_def_8', 'dis_def_8', 'o_def_8', 'dir_def_8', 'distance_def_8', 
#def player 9
'x_def_9', 'y_def_9', 'ahead_of_pass_catcher_def_9', 's_def_9', 
'a_def_9', 'dis_def_9', 'o_def_9', 'dir_def_9', 'distance_def_9', 
#def player 10
'x_def_10', 'y_def_10', 'ahead_of_pass_catcher_def_10', 's_def_10', 
'a_def_10', 'dis_def_10', 'o_def_10', 'dir_def_10', 'distance_def_10', 
#def player 11
'x_def_11', 'y_def_11', 'ahead_of_pass_catcher_def_11', 's_def_11', 
'a_def_11', 'dis_def_11', 'o_def_11', 'dir_def_11', 'distance_def_11',
#outcome
'outcome_distance']].apply(pd.to_numeric)

#remove plays where pass was caught by QB                             
data = data[data['position_pass_catcher'] != 'QB']
                             
#create dummy variables from player position (only pass catcher and defensive players)
dummies = pd.get_dummies(data[['position_pass_catcher', 'position_def_1', 'position_def_2', 'position_def_3', 'position_def_4', 'position_def_5']], drop_first=True)                        
data = data.join(dummies)

#split into training and test by holding out these games for test data
test_games = [2022110700,2022110610,2022110609,2022110608,2022110607
,2022110606,2022110605,2022110604,2022110603,2022110602,2022110601
,2022110600,2022110300,2022103100,2022103012,2022103011,2022103010
,2022103009,2022103008,2022103007,2022103006,2022103005,2022103004
,2022103003,2022103002]

#almost 1000 plays for testing and the rest for training
train = data[~(data['gameId'].isin(test_games))].copy()
test = data[(data['gameId'].isin(test_games))].copy()

#these features provides the most accurate results
features = ['s_pass_catcher', 'a_pass_catcher', 'dir_pass_catcher', 'pass_catcher_dir_change', 'pass_catcher_o_change', 'pass_catcher_s_change',
       'catch_distance_from_sideline', 'catch_distance_from_los',  'y_def_1', 's_def_1',
       'a_def_1', 'distance_def_1', 'o_def_1', 'dir_def_1', 'y_def_2',
       's_def_2', 'a_def_2', 'distance_def_2', 'o_def_2', 'dir_def_2',
       'y_def_3', 's_def_3', 'a_def_3', 'distance_def_3', 'o_def_3',
       'dir_def_3', 'y_def_4', 's_def_4', 'a_def_4', 'distance_def_4',
       'o_def_4', 'dir_def_4', 'off_players_ahead_of_pass_catcher',
       'def_players_ahead_of_pass_catcher', 'yardsToGo']

#import random forest and grid search
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

#run grid search to find best paramters
grid = GridSearchCV(RandomForestRegressor(random_state=0),
                    param_grid={
                        "n_estimators": [25, 50, 200],
                        "criterion": ['squared_error'],
                        "max_depth": [3,5,7],
                        "max_features": ["log2", "sqrt"]},
                    scoring='r2')
#fit to the training data
grid.fit(train[features],train['outcome_distance'])

#find parameters
grid.best_params_
#view r2
grid.best_score_

#view accuracy on test data
from sklearn.metrics import r2_score
test['predicted_yards'] = grid.predict(test[features])
test['predicted_yards'].corr(test['outcome_distance'])
r2_score(test["outcome_distance"], test['predicted_yards'])

# get predictions on full data
data['predicted_yards'] = grid.predict(data[features])
data['predicted_yards'].corr(data['outcome_distance'])

#manually adjust predicted values that are greater than catch location and goal line distance
data['predicted_yards'] = np.where(data['predicted_yards'] > data['catch_distance_from_endzone'], data['catch_distance_from_endzone'], data['predicted_yards'])

#calculated yards over and under expected
data['yards_under_expected'] =  data['predicted_yards'] - data['outcome_distance']
data['yards_over_expected'] =   data['outcome_distance'] - data['predicted_yards']

#yards saved by defense
yards_under_expected = data[['defense_team', 'yards_under_expected']].rename(columns={'defense_team': 'team'})
team_yards_under_expected = yards_under_expected.groupby('team')['yards_under_expected'].agg(['count', 'sum', 'mean']).sort_values('mean', ascending=False).reset_index().rename(columns={'count': 'total_plays', 'sum': 'total_yards_under_expected', 'mean': 'avg_yards_under_expected'})

#yards over expected by pass catchers
yards_over_expected = data[['player_pass_catcher', 'position_pass_catcher', 'yards_over_expected']].rename(columns={'player_pass_catcher': 'player', 'position_pass_catcher': 'position'})
player_yards_over_expected = yards_over_expected.groupby(['player', 'position'])['yards_over_expected'].agg(['count', 'sum', 'mean']).sort_values('mean', ascending=False).reset_index().rename(columns={'count': 'total_plays', 'sum': 'total_yards_over_expected', 'mean': 'avg_yards_over_expected'})
player_yards_over_expected = player_yards_over_expected[player_yards_over_expected['total_plays'] >= 25]


#example plays
example_play1 = data[data['playDescription'] == '(12:12) (Shotgun) K.Murray pass short left to R.Moore to CAR 21 for 6 yards (F.Luvu).']
example_play2 = data[data['playDescription'] == '(4:22) (Shotgun) A.Dalton pass short right to J.Johnson for 41 yards, TOUCHDOWN. Pass 11, YAC 30']


import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_football_field():
    
    # create rectangle
    rect = patches.Rectangle((0, 0), 140, 53.3, color='#274c43', zorder=0)

    # create subplot for field
    fig, ax = plt.subplots(1, figsize=(12, 6.33), facecolor = "black")

    # Add rectangle
    ax.add_patch(rect)

    # plot outside rectangle of field
    plt.plot([120.5, -0.5, -0.5, 120.5, 120.5],
             [53.3, 53.3, 0, 0, 53.3],
             color='white',linewidth=3.5, zorder = 10)
    
    #add lines for endzone and every 5 yards
    plt.plot([10, 10, 15, 15,  20, 20, 25, 25, 30, 30, 35,35, 40, 40, 45, 45, 50, 50, 55, 55, 60, 60, 65, 65, 70, 70, 75, 75, 80,
              80, 85,85, 90, 90,  95,95, 100, 100, 105,105, 110, 110],
             [0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 
              0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 
               0, 0, 53.3, 53.3, 0, 0, 53.3],
             color='white', alpha=0.3, zorder = 0)

    # create left endzone
    left_end_zone = patches.Rectangle((0, 0), 10, 53.3, facecolor='#274c43',  zorder=0)

    # create right endzone
    right_end_zone = patches.Rectangle((110, 0), 120, 53.3, facecolor='#274c43',  zorder=0)

    # add endzones to plot
    ax.add_patch(left_end_zone)
    ax.add_patch(right_end_zone)

    # set the limits of x-axis
    plt.xlim(-1, 121)

    # set the limits of y-axis
    plt.ylim(-1, 54.3)

    # remove axis
    plt.axis('off')

    # plot yard line numbers from x = 20 and ending at x = 110 
    for x in range(20, 110, 10):

        # initialize  number variable
        number = x

        # if x > 50, subtract it from 120
        if x > 50:
            number = 120 - x

        # plot the text at the bottom
        plt.text(x, 5, str(number - 10),
                 horizontalalignment='center',
                 fontsize=20,
                 color='white')

        # plot the text at the top
        plt.text(x - 0.95, 53.3 - 5, str(number - 10),
                 horizontalalignment='center',
                 fontsize=20,
                 color='white',
                 rotation=180)

    # make ground markings
    for x in range(11, 110):
            ax.plot([x, x], [0.4, 0.7], color='white', zorder = 0)
            ax.plot([x, x], [53.0, 52.5], color='white', zorder = 0)
            ax.plot([x, x], [22.91, 23.57], color='white', alpha=0.1, zorder = 0)
            ax.plot([x, x], [29.73, 30.39], color='white', alpha=0.1, zorder = 0)

    # Returning the figure and axis
    return fig, ax

#play1
fig, ax = create_football_field()
ax.scatter(example_play1['x_pass_catcher'], example_play1['y_pass_catcher'], facecolor = '#AF983F', edgecolors='black', s= 2**9,  zorder=10)
ax.scatter(example_play1['x_off_1'], example_play1['y_off_1'], facecolor = 'white', edgecolors='black', s= 2**9, zorder=10)
ax.scatter(example_play1['x_off_2'], example_play1['y_off_2'], facecolor = 'white', edgecolors='black', s= 2**9, zorder=10)
ax.scatter(example_play1['x_off_3'], example_play1['y_off_3'], facecolor = 'white', edgecolors='black', s= 2**9, zorder=10)
ax.scatter(example_play1['x_off_4'], example_play1['y_off_4'], facecolor = 'white', edgecolors='black', s= 2**9, zorder=10)
ax.scatter(example_play1['x_off_5'], example_play1['y_off_5'], facecolor = 'white', edgecolors='black', s= 2**9, zorder=10)
ax.scatter(example_play1['x_off_6'], example_play1['y_off_6'], facecolor = 'white', edgecolors='black', s= 2**9, zorder=10)
ax.scatter(example_play1['x_off_7'], example_play1['y_off_7'], facecolor = 'white', edgecolors='black', s= 2**9, zorder=10)
ax.scatter(example_play1['x_off_8'], example_play1['y_off_8'], facecolor = 'white', edgecolors='black', s= 2**9, zorder=10)
ax.scatter(example_play1['x_off_9'], example_play1['y_off_9'], facecolor = 'white', edgecolors='black', s= 2**9, zorder=10)
ax.scatter(example_play1['x_off_10'], example_play1['y_off_10'], facecolor = 'white', edgecolors='black', s= 2**9, zorder=10)

ax.scatter(example_play1['x_def_1'], example_play1['y_def_1'], facecolor = 'black', edgecolors='white', s= 2**9, zorder=10)
ax.scatter(example_play1['x_def_2'], example_play1['y_def_2'], facecolor = 'black', edgecolors='white', s= 2**9, zorder=10)
ax.scatter(example_play1['x_def_3'], example_play1['y_def_3'], facecolor = 'black', edgecolors='white', s= 2**9, zorder=10)
ax.scatter(example_play1['x_def_4'], example_play1['y_def_4'], facecolor = 'black', edgecolors='white', s= 2**9, zorder=10)
ax.scatter(example_play1['x_def_5'], example_play1['y_def_5'], facecolor = 'black', edgecolors='white', s= 2**9, zorder=10)
ax.scatter(example_play1['x_def_6'], example_play1['y_def_6'], facecolor = 'black', edgecolors='white', s= 2**9, zorder=10)
ax.scatter(example_play1['x_def_7'], example_play1['y_def_7'], facecolor = 'black', edgecolors='white', s= 2**9, zorder=10)
ax.scatter(example_play1['x_def_8'], example_play1['y_def_8'], facecolor = 'black', edgecolors='white', s= 2**9, zorder=10)
ax.scatter(example_play1['x_def_9'], example_play1['y_def_9'], facecolor = 'black', edgecolors='white', s= 2**9, zorder=10)
ax.scatter(example_play1['x_def_10'], example_play1['y_def_10'], facecolor = 'black', edgecolors='white', s= 2**9, zorder=10)
ax.scatter(example_play1['x_def_11'], example_play1['y_def_11'], facecolor = 'black', edgecolors='white', s= 2**9, zorder=10)
#ax.annotate('Expected Yards: ' + round(example_play1['predicted_yards'].values[0],2).astype(str) , xy = (10,55))
ax.annotate(xy=(example_play1['x_pass_catcher'].values[0], example_play1['y_pass_catcher'].values[0]), text=example_play1['position_pass_catcher'].values[0], color="black", weight='bold', ha= 'center', va='center', fontsize=8, zorder=10)
ax.annotate(xy=(example_play1['x_off_1'].values[0], example_play1['y_off_1'].values[0]), text=example_play1['position_off_1'].values[0], color="black", weight='bold', ha= 'center', va='center', fontsize=8, zorder=10)
ax.annotate(xy=(example_play1['x_off_2'].values[0], example_play1['y_off_2'].values[0]), text=example_play1['position_off_2'].values[0], color="black", weight='bold', ha= 'center', va='center',fontsize=8, zorder=10)
ax.annotate(xy=(example_play1['x_off_3'].values[0], example_play1['y_off_3'].values[0]), text=example_play1['position_off_3'].values[0], color="black", weight='bold', ha= 'center', va='center', fontsize=8, zorder=10)
ax.annotate(xy=(example_play1['x_off_4'].values[0], example_play1['y_off_4'].values[0]), text=example_play1['position_off_4'].values[0], color="black", weight='bold', ha= 'center', va='center', fontsize=8,zorder=10)
ax.annotate(xy=(example_play1['x_off_5'].values[0], example_play1['y_off_5'].values[0]), text=example_play1['position_off_5'].values[0], color="black", weight='bold', ha= 'center', va='center', fontsize=8,zorder=10)
ax.annotate(xy=(example_play1['x_off_6'].values[0], example_play1['y_off_6'].values[0]), text=example_play1['position_off_6'].values[0], color="black", weight='bold', ha= 'center', va='center', fontsize=8,zorder=10)
ax.annotate(xy=(example_play1['x_off_7'].values[0], example_play1['y_off_7'].values[0]), text=example_play1['position_off_7'].values[0], color="black", weight='bold', ha= 'center', va='center', fontsize=8,zorder=10)
ax.annotate(xy=(example_play1['x_off_8'].values[0], example_play1['y_off_8'].values[0]), text=example_play1['position_off_8'].values[0], color="black", weight='bold', ha= 'center', va='center', fontsize=8,zorder=10)
ax.annotate(xy=(example_play1['x_off_9'].values[0], example_play1['y_off_9'].values[0]), text=example_play1['position_off_9'].values[0], color="black", weight='bold', ha= 'center', va='center', fontsize=8,zorder=10)
ax.annotate(xy=(example_play1['x_off_10'].values[0], example_play1['y_off_10'].values[0]), text=example_play1['position_off_10'].values[0], color="black", weight='bold', ha= 'center', va='center', fontsize=8,zorder=10)

ax.annotate(xy=(example_play1['x_def_1'].values[0], example_play1['y_def_1'].values[0]), text=example_play1['position_def_1'].values[0], color="white", weight='bold', ha= 'center', va='center', fontsize=8,zorder=10)
ax.annotate(xy=(example_play1['x_def_2'].values[0], example_play1['y_def_2'].values[0]), text=example_play1['position_def_2'].values[0], color="white", weight='bold', ha= 'center', va='center', fontsize=8,zorder=10)
ax.annotate(xy=(example_play1['x_def_3'].values[0], example_play1['y_def_3'].values[0]), text=example_play1['position_def_3'].values[0], color="white", weight='bold', ha= 'center', va='center', fontsize=8,zorder=10)
ax.annotate(xy=(example_play1['x_def_4'].values[0], example_play1['y_def_4'].values[0]), text=example_play1['position_def_4'].values[0], color="white", weight='bold', ha= 'center', va='center', fontsize=8,zorder=10)
ax.annotate(xy=(example_play1['x_def_5'].values[0], example_play1['y_def_5'].values[0]), text=example_play1['position_def_5'].values[0], color="white", weight='bold', ha= 'center', va='center', fontsize=8,zorder=10)
ax.annotate(xy=(example_play1['x_def_6'].values[0], example_play1['y_def_6'].values[0]), text=example_play1['position_def_6'].values[0], color="white", weight='bold', ha= 'center', va='center', fontsize=8,zorder=10)
ax.annotate(xy=(example_play1['x_def_7'].values[0], example_play1['y_def_7'].values[0]), text=example_play1['position_def_7'].values[0], color="white", weight='bold', ha= 'center', va='center', fontsize=8,zorder=10)
ax.annotate(xy=(example_play1['x_def_8'].values[0], example_play1['y_def_8'].values[0]), text=example_play1['position_def_8'].values[0], color="white", weight='bold', ha= 'center', va='center', fontsize=8,zorder=10)
ax.annotate(xy=(example_play1['x_def_9'].values[0], example_play1['y_def_9'].values[0]), text=example_play1['position_def_9'].values[0], color="white", weight='bold', ha= 'center', va='center', fontsize=8,zorder=10)
ax.annotate(xy=(example_play1['x_def_10'].values[0], example_play1['y_def_10'].values[0]), text=example_play1['position_def_10'].values[0], color="white", weight='bold', ha= 'center', va='center', fontsize=8,zorder=10)
ax.annotate(xy=(example_play1['x_def_11'].values[0], example_play1['y_def_11'].values[0]), text=example_play1['position_def_11'].values[0], color="white", weight='bold', ha= 'center', va='center', fontsize=8,zorder=10)
plt.show()

#play2
fig, ax = create_football_field()
ax.scatter(example_play2['x_pass_catcher'], example_play2['y_pass_catcher'], facecolor = '#AF983F', edgecolors='black', s= 2**9,  zorder=10)
ax.scatter(example_play2['x_off_1'], example_play2['y_off_1'], facecolor = 'white', edgecolors='black', s= 2**9, zorder=10)
ax.scatter(example_play2['x_off_2'], example_play2['y_off_2'], facecolor = 'white', edgecolors='black', s= 2**9, zorder=10)
ax.scatter(example_play2['x_off_3'], example_play2['y_off_3'], facecolor = 'white', edgecolors='black', s= 2**9, zorder=10)
ax.scatter(example_play2['x_off_4'], example_play2['y_off_4'], facecolor = 'white', edgecolors='black', s= 2**9, zorder=10)
ax.scatter(example_play2['x_off_5'], example_play2['y_off_5'], facecolor = 'white', edgecolors='black', s= 2**9, zorder=10)
ax.scatter(example_play2['x_off_6'], example_play2['y_off_6'], facecolor = 'white', edgecolors='black', s= 2**9, zorder=10)
ax.scatter(example_play2['x_off_7'], example_play2['y_off_7'], facecolor = 'white', edgecolors='black', s= 2**9, zorder=10)
ax.scatter(example_play2['x_off_8'], example_play2['y_off_8'], facecolor = 'white', edgecolors='black', s= 2**9, zorder=10)
ax.scatter(example_play2['x_off_9'], example_play2['y_off_9'], facecolor = 'white', edgecolors='black', s= 2**9, zorder=10)
ax.scatter(example_play2['x_off_10'], example_play2['y_off_10'], facecolor = 'white', edgecolors='black', s= 2**9, zorder=10)

ax.scatter(example_play2['x_def_1'], example_play2['y_def_1'], facecolor = 'black', edgecolors='white', s= 2**9, zorder=10)
ax.scatter(example_play2['x_def_2'], example_play2['y_def_2'], facecolor = 'black', edgecolors='white', s= 2**9, zorder=10)
ax.scatter(example_play2['x_def_3'], example_play2['y_def_3'], facecolor = 'black', edgecolors='white', s= 2**9, zorder=10)
ax.scatter(example_play2['x_def_4'], example_play2['y_def_4'], facecolor = 'black', edgecolors='white', s= 2**9, zorder=10)
ax.scatter(example_play2['x_def_5'], example_play2['y_def_5'], facecolor = 'black', edgecolors='white', s= 2**9, zorder=10)
ax.scatter(example_play2['x_def_6'], example_play2['y_def_6'], facecolor = 'black', edgecolors='white', s= 2**9, zorder=10)
ax.scatter(example_play2['x_def_7'], example_play2['y_def_7'], facecolor = 'black', edgecolors='white', s= 2**9, zorder=10)
ax.scatter(example_play2['x_def_8'], example_play2['y_def_8'], facecolor = 'black', edgecolors='white', s= 2**9, zorder=10)
ax.scatter(example_play2['x_def_9'], example_play2['y_def_9'], facecolor = 'black', edgecolors='white', s= 2**9, zorder=10)
ax.scatter(example_play2['x_def_10'], example_play2['y_def_10'], facecolor = 'black', edgecolors='white', s= 2**9, zorder=10)
ax.scatter(example_play2['x_def_11'], example_play2['y_def_11'], facecolor = 'black', edgecolors='white', s= 2**9, zorder=10)
#ax.annotate('Expected Yards: ' + round(example_play2['predicted_yards'].values[0],2).astype(str) , xy = (10,55))
ax.annotate(xy=(example_play2['x_pass_catcher'].values[0], example_play2['y_pass_catcher'].values[0]), text=example_play2['position_pass_catcher'].values[0], color="black", weight='bold', ha= 'center', va='center', fontsize=8, zorder=10)
ax.annotate(xy=(example_play2['x_off_1'].values[0], example_play2['y_off_1'].values[0]), text=example_play2['position_off_1'].values[0], color="black", weight='bold', ha= 'center', va='center', fontsize=8, zorder=10)
ax.annotate(xy=(example_play2['x_off_2'].values[0], example_play2['y_off_2'].values[0]), text=example_play2['position_off_2'].values[0], color="black", weight='bold', ha= 'center', va='center',fontsize=8, zorder=10)
ax.annotate(xy=(example_play2['x_off_3'].values[0], example_play2['y_off_3'].values[0]), text=example_play2['position_off_3'].values[0], color="black", weight='bold', ha= 'center', va='center', fontsize=8, zorder=10)
ax.annotate(xy=(example_play2['x_off_4'].values[0], example_play2['y_off_4'].values[0]), text=example_play2['position_off_4'].values[0], color="black", weight='bold', ha= 'center', va='center', fontsize=8,zorder=10)
ax.annotate(xy=(example_play2['x_off_5'].values[0], example_play2['y_off_5'].values[0]), text=example_play2['position_off_5'].values[0], color="black", weight='bold', ha= 'center', va='center', fontsize=8,zorder=10)
ax.annotate(xy=(example_play2['x_off_6'].values[0], example_play2['y_off_6'].values[0]), text=example_play2['position_off_6'].values[0], color="black", weight='bold', ha= 'center', va='center', fontsize=8,zorder=10)
ax.annotate(xy=(example_play2['x_off_7'].values[0], example_play2['y_off_7'].values[0]), text=example_play2['position_off_7'].values[0], color="black", weight='bold', ha= 'center', va='center', fontsize=8,zorder=10)
ax.annotate(xy=(example_play2['x_off_8'].values[0], example_play2['y_off_8'].values[0]), text=example_play2['position_off_8'].values[0], color="black", weight='bold', ha= 'center', va='center', fontsize=8,zorder=10)
ax.annotate(xy=(example_play2['x_off_9'].values[0], example_play2['y_off_9'].values[0]), text=example_play2['position_off_9'].values[0], color="black", weight='bold', ha= 'center', va='center', fontsize=8,zorder=10)
ax.annotate(xy=(example_play2['x_off_10'].values[0], example_play2['y_off_10'].values[0]), text=example_play2['position_off_10'].values[0], color="black", weight='bold', ha= 'center', va='center', fontsize=8,zorder=10)

ax.annotate(xy=(example_play2['x_def_1'].values[0], example_play2['y_def_1'].values[0]), text=example_play2['position_def_1'].values[0], color="white", weight='bold', ha= 'center', va='center', fontsize=8,zorder=10)
ax.annotate(xy=(example_play2['x_def_2'].values[0], example_play2['y_def_2'].values[0]), text=example_play2['position_def_2'].values[0], color="white", weight='bold', ha= 'center', va='center', fontsize=8,zorder=10)
ax.annotate(xy=(example_play2['x_def_3'].values[0], example_play2['y_def_3'].values[0]), text=example_play2['position_def_3'].values[0], color="white", weight='bold', ha= 'center', va='center', fontsize=8,zorder=10)
ax.annotate(xy=(example_play2['x_def_4'].values[0], example_play2['y_def_4'].values[0]), text=example_play2['position_def_4'].values[0], color="white", weight='bold', ha= 'center', va='center', fontsize=8,zorder=10)
ax.annotate(xy=(example_play2['x_def_5'].values[0], example_play2['y_def_5'].values[0]), text=example_play2['position_def_5'].values[0], color="white", weight='bold', ha= 'center', va='center', fontsize=8,zorder=10)
ax.annotate(xy=(example_play2['x_def_6'].values[0], example_play2['y_def_6'].values[0]), text=example_play2['position_def_6'].values[0], color="white", weight='bold', ha= 'center', va='center', fontsize=8,zorder=10)
ax.annotate(xy=(example_play2['x_def_7'].values[0], example_play2['y_def_7'].values[0]), text=example_play2['position_def_7'].values[0], color="white", weight='bold', ha= 'center', va='center', fontsize=8,zorder=10)
ax.annotate(xy=(example_play2['x_def_8'].values[0], example_play2['y_def_8'].values[0]), text=example_play2['position_def_8'].values[0], color="white", weight='bold', ha= 'center', va='center', fontsize=8,zorder=10)
ax.annotate(xy=(example_play2['x_def_9'].values[0], example_play2['y_def_9'].values[0]), text=example_play2['position_def_9'].values[0], color="white", weight='bold', ha= 'center', va='center', fontsize=8,zorder=10)
ax.annotate(xy=(example_play2['x_def_10'].values[0], example_play2['y_def_10'].values[0]), text=example_play2['position_def_10'].values[0], color="white", weight='bold', ha= 'center', va='center', fontsize=8,zorder=10)
ax.annotate(xy=(example_play2['x_def_11'].values[0], example_play2['y_def_11'].values[0]), text=example_play2['position_def_11'].values[0], color="white", weight='bold', ha= 'center', va='center', fontsize=8,zorder=10)

plt.show()