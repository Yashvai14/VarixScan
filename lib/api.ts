// api.ts
import axios, { AxiosInstance, AxiosResponse } from "axios";

// Base URL of your Render backend
const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "https://varixscan.onrender.com";

// Create an axios instance
const api: AxiosInstance = axios.create({
  baseURL: BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

// Generic GET request
export const getRequest = async <T>(endpoint: string, params?: Record<string, unknown>): Promise<T> => {
  const response: AxiosResponse<T> = await api.get(endpoint, { params });
  return response.data;
};

// Generic POST request
export const postRequest = async <T>(endpoint: string, data?: Record<string, unknown>): Promise<T> => {
  const response: AxiosResponse<T> = await api.post(endpoint, data);
  return response.data;
};

// Generic PUT request
export const putRequest = async <T>(endpoint: string, data?: Record<string, unknown>): Promise<T> => {
  const response: AxiosResponse<T> = await api.put(endpoint, data);
  return response.data;
};

// Generic DELETE request
export const deleteRequest = async <T>(endpoint: string): Promise<T> => {
  const response: AxiosResponse<T> = await api.delete(endpoint);
  return response.data;
};

export default api;
