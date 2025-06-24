import { useState } from 'react'
import './assets/css/styles.css'
import './global.css'
import Home from './assets/pages/Home.jsx'
import History from './assets/pages/History.jsx'
import Information from './assets/pages/Information.jsx'
import Support from './assets/pages/Support.jsx'
import LogIn from './assets/pages/LogIn.jsx'
import BuyTokens from './assets/pages/BuyTokens.jsx'
import Register from './assets/pages/Register.jsx'
import ForgotPass from './assets/pages/ForgotPass.jsx'
import Profile from './assets/pages/Profile.jsx'
import AdminTokenRequests from './assets/admin/AdminTokenRequests.jsx'
import ContactManagement from './assets/admin/ContactManagement.jsx'
import EmployeeManagement from './assets/admin/EmployeeManagement.jsx'
import SupportManagement from './assets/admin/SupportManagement.jsx'
import UserManager from './assets/admin/UserManager.jsx'
import PaymentManagement from './assets/admin/PaymentManagement.jsx'
import NotFound from './assets/pages/NotFound.jsx'
import { Routes, Route } from 'react-router-dom'
import { LoadingProvider } from './assets/utils/LoadingContext.jsx'
import RequireAdmin from './assets/utils/RequireAdmin.jsx'

function App() {
  return (
    <LoadingProvider>
      <div className="App">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/History" element={<History />} />
          <Route path="/Information" element={<Information />} />
          <Route path="/Support" element={<Support />} />
          <Route path="/LogIn" element={<LogIn />} />
          <Route path="/BuyTokens" element={<BuyTokens />} />
          <Route path="/Register" element={<Register />} />
          <Route path="/ForgotPass" element={<ForgotPass />} />
          <Route path="/Profile" element={<Profile />} />
          <Route
            path="/AdminTokenRequests"
            element={
              <RequireAdmin>
                <AdminTokenRequests />
              </RequireAdmin>
            }
          />
          <Route
            path="/ContactManagement"
            element={
              <RequireAdmin>
                <ContactManagement />
              </RequireAdmin>
            }
          />
          <Route
            path="/EmployeeManagement"
            element={
              <RequireAdmin>
                <EmployeeManagement />
              </RequireAdmin>
            }
          />
          <Route
            path="/SupportManagement"
            element={
              <RequireAdmin>
                <SupportManagement />
              </RequireAdmin>
            }
          />
          <Route
            path="/UserManager"
            element={
              <RequireAdmin>
                <UserManager />
              </RequireAdmin>
            }
          />
          <Route
            path="/PaymentManagement"
            element={
              <RequireAdmin>
                <PaymentManagement />
              </RequireAdmin>
            }
          />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </div>
    </LoadingProvider>
  )
}

export default App
