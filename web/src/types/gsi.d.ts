// Google Identity Services（https://accounts.google.com/gsi/client）用到的部分
interface GoogleCredentialResponse {
  credential: string
}

interface Window {
  google?: {
    accounts: {
      id: {
        initialize(options: {
          client_id: string
          callback: (response: GoogleCredentialResponse) => void
          ux_mode?: "popup" | "redirect"
          auto_select?: boolean
          use_fedcm_for_button?: boolean
        }): void
        renderButton(
          parent: HTMLElement,
          options: { theme?: string; size?: string; text?: string; shape?: string; locale?: string; width?: number },
        ): void
        disableAutoSelect(): void
      }
    }
  }
}
