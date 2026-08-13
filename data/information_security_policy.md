# Information Security Policy

This policy defines the minimum security standards every employee, contractor, and third-party user must follow when accessing company systems, data, and networks. Failure to comply may result in disciplinary action, contract termination, or legal liability.

## Classification of Information

All company information is assigned one of four classification levels:

**Public** — Information approved for unrestricted external distribution. Press releases, published job adverts, and marketing materials are examples. No special handling required.

**Internal** — Information intended for employees only. Internal memos, process documentation, and project plans fall here. Must not be shared publicly or with third parties without approval.

**Confidential** — Sensitive business information whose disclosure could harm the company or individuals. Financial reports, employee salary data, customer contracts, and product roadmaps are confidential. Must be encrypted at rest and in transit, accessible only to those with a documented business need.

**Restricted** — The highest classification. Personally identifiable information (PII) subject to GDPR, trade secrets, authentication credentials, and cryptographic keys. Access is granted individually, logged, and reviewed quarterly. Must be encrypted with AES-256 or equivalent.

When in doubt about classification, treat information as Confidential until confirmed otherwise.

## Passwords and Authentication

Every account used to access company resources must use a strong password. A strong password is at least 16 characters long and contains a mix of upper and lower case letters, numbers, and symbols. Passwords must not be dictionary words, names, or predictable sequences such as `Password1!`.

Employees must not reuse passwords across company and personal accounts. The company provides a licensed password manager; all employees are required to use it for company credentials. Passwords must never be stored in plain text files, spreadsheets, or shared via email.

Multi-factor authentication (MFA) is mandatory for all accounts with access to Confidential or Restricted data, all cloud-hosted services, and all VPN connections. The preferred second factor is a hardware token or authenticator app. SMS-based MFA is permitted only where no alternative exists.

Shared accounts are prohibited. Every person accessing a system must have an individual account so that access can be traced, revoked, and audited individually.

## Device Security

### Company-Issued Devices

All laptops, desktops, and mobile devices issued by the company must:

- Run only approved, up-to-date operating systems. Devices running end-of-life operating systems must be reported to IT immediately.
- Have full-disk encryption enabled (FileVault on macOS, BitLocker on Windows).
- Run the company-mandated endpoint protection (antivirus and EDR) at all times. Disabling or circumventing this software is strictly prohibited.
- Apply operating system and application security patches within 7 days of release for critical patches, and within 30 days for all others.
- Lock automatically after 5 minutes of inactivity and require a password to unlock.

Employees must not install software not approved by IT on company devices. The IT department maintains a list of approved applications and a request process for additions.

### Personal Devices (BYOD)

Personal devices may access company email and calendar through the company-approved mobile device management (MDM) profile. Installing the MDM profile implies consent to remote wipe of company data from that device if it is lost or stolen. Personal devices may not store Confidential or Restricted company data locally.

### Lost or Stolen Devices

A lost or stolen company device must be reported to IT and the employee's manager within two hours of discovery. IT will remotely wipe the device and revoke its credentials. The employee must also file a police report if theft is suspected.

## Network Security

Employees must use the company VPN whenever accessing Internal, Confidential, or Restricted systems from outside the office. This applies to home networks and is mandatory on public Wi-Fi such as in cafés, airports, and hotels.

Personal hotspots are permitted as an alternative to public Wi-Fi and do not require VPN provided the hotspot is password-protected and controlled by the employee.

Employees must not connect untrusted or unknown devices to the corporate network, including USB drives received from unknown sources. Physical media found in common areas (car parks, reception) must be handed to IT without being plugged in — this is a common social engineering vector.

## Data Handling and Storage

Confidential and Restricted data must be stored only in company-approved locations: the corporate cloud storage (SharePoint/OneDrive), the company's on-premise file servers, or encrypted databases. Storing such data on personal cloud accounts (Google Drive, Dropbox personal, iCloud) is prohibited.

Before disposing of any device — including old hard drives, USB sticks, or phones — employees must submit it to IT for secure erasure. Employees must not dispose of devices themselves, even recycling old personal hardware that was used for company work.

Hard-copy documents containing Confidential or Restricted information must be shredded, not placed in general waste bins. Each office has a dedicated confidential waste bin for this purpose.

## Acceptable Use

Company systems are provided for business purposes. Incidental personal use (checking personal email, brief personal browsing) is tolerated but must not interfere with work, consume significant bandwidth, or expose company systems to risk.

The following uses are prohibited on all company systems and networks:

- Accessing, downloading, or distributing illegal content of any kind
- Attempting to access systems or data beyond one's authorised scope ("hacking")
- Mining cryptocurrency
- Running personal servers or hosting personal services
- Circumventing security controls, including disabling antivirus, using anonymising proxies to bypass web filters, or exploiting system vulnerabilities

## Phishing and Social Engineering

Employees must be vigilant against phishing emails, vishing (phone-based fraud), and other social engineering attacks. If an email requests credentials, financial transfers, or sensitive data — even if it appears to come from a colleague or senior executive — employees should verify the request through a separate channel (a phone call, in person) before complying.

Suspicious emails should be reported using the "Report Phishing" button in the email client, or forwarded to security@company.com. Do not click links or open attachments in suspected phishing emails.

The company conducts simulated phishing exercises periodically. Employees who repeatedly fail simulations will be required to complete additional security awareness training.

## Incident Response

A security incident is any event that compromises or may compromise the confidentiality, integrity, or availability of company information or systems. Examples include a suspected malware infection, accidental disclosure of Restricted data, an unrecognised login to your account, or a ransomware pop-up.

Any suspected security incident must be reported to IT (security@company.com or the IT hotline) immediately — do not wait to investigate yourself, as this may destroy forensic evidence. The IT security team will assess, contain, and investigate the incident. Employees must cooperate fully with the investigation.

## Third-Party Access

Vendors, contractors, and partners who require access to company systems must sign a Data Processing Agreement before access is granted. They are bound by the same security standards as employees and may access only the minimum systems and data necessary for their work. Third-party access is reviewed and renewed annually.

## Review

This policy is reviewed annually by the IT Security team and updated as necessary. Employees are notified of material changes. Questions about this policy should be directed to security@company.com.
