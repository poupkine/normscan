import { configureStore } from '@reduxjs/toolkit';
import errorReducer from './slices/errorSlice';
import pagesReducer from '@pages/slice';

export const store = configureStore({
  reducer: {
    error: errorReducer,
    pages: pagesReducer
  }
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
export type AppStore = typeof store;