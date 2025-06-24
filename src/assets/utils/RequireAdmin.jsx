// src/assets/utils/RequireAdmin.jsx
import React from "react";
import { Navigate } from "react-router-dom";

const RequireAdmin = ({ children }) => {
    const token = localStorage.getItem("token");
    const role = localStorage.getItem("role");

    if (!token || role !== "admin") {
        return <Navigate to="/NotFound" replace />;
    }

    return children;
};

export default RequireAdmin;
