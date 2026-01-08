import flet as ft


def main(page: ft.Page):
    page.title = "Itadaku - Flet Sample App"
    page.window.width = 400
    page.window.height = 500
    page.window.resizable = True
    page.theme_mode = ft.ThemeMode.LIGHT

    # タイトル
    title = ft.Text(
        "いただく管理アプリ",
        size=28,
        weight=ft.FontWeight.BOLD,
        color=ft.Colors.BLUE_700,
    )

    # カウンター表示
    counter_display = ft.Text(
        "0",
        size=48,
        weight=ft.FontWeight.BOLD,
        color=ft.Colors.BLUE,
    )

    # カウンター状態
    counter = {"value": 0}

    # ボタンのクリックイベント
    def increment(e):
        counter["value"] += 1
        counter_display.value = str(counter["value"])
        page.update()

    def decrement(e):
        counter["value"] -= 1
        counter_display.value = str(counter["value"])
        page.update()

    def reset(e):
        counter["value"] = 0
        counter_display.value = "0"
        page.update()

    # ボタン
    btn_increment = ft.Button(
        "増加",
        on_click=increment,
        width=100,
        color=ft.Colors.WHITE,
        bgcolor=ft.Colors.GREEN_700,
    )

    btn_decrement = ft.Button(
        "減少",
        on_click=decrement,
        width=100,
        color=ft.Colors.WHITE,
        bgcolor=ft.Colors.RED_700,
    )

    btn_reset = ft.Button(
        "リセット",
        on_click=reset,
        width=100,
        color=ft.Colors.WHITE,
        bgcolor=ft.Colors.GREY_700,
    )

    # ボタンレイアウト
    button_row = ft.Row(
        controls=[btn_decrement, btn_increment, btn_reset],
        alignment=ft.MainAxisAlignment.CENTER,
        spacing=10,
    )

    # メインコンテナ
    container = ft.Container(
        content=ft.Column(
            controls=[
                title,
                ft.Divider(),
                ft.Text("カウンター", size=18, weight=ft.FontWeight.W_600),
                counter_display,
                button_row,
                ft.Divider(),
                ft.Text(
                    "これはFletを使ったサンプルアプリケーションです。",
                    size=12,
                    color=ft.Colors.GREY_700,
                ),
            ],
            alignment=ft.MainAxisAlignment.START,
            spacing=20,
        ),
        padding=30,
    )

    page.add(container)


if __name__ == "__main__":
    ft.run(main)
