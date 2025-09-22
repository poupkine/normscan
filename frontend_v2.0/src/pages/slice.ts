import { combineReducers } from '@reduxjs/toolkit';
import homePageReducer from './home/slice';

const pagesReducer = combineReducers({
  homePage: homePageReducer
});

export default pagesReducer;