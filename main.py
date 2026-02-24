import flet as ft
import threading
from pathlib import Path
from sample import Translator

def main(page: ft.Page):
    page.title = "Itadaku - 翻訳アプリ"
    page.window.width = 600
    page.window.height = 700
    page.theme_mode = ft.ThemeMode.LIGHT

    # 状態管理
    state = {
        "translator": None,
        "is_loading": True,
        "is_translating": False,
    }

    # UIコンポーネント
    title = ft.Text("AI 翻訳", size=28, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_700)
    
    status_text = ft.Text("モデルを読み込んでいます...", color=ft.Colors.ORANGE_700)
    progress_ring = ft.ProgressRing(width=20, height=20, stroke_width=2)
    status_row = ft.Row([progress_ring, status_text], visible=True)

    source_lang_dropdown = ft.Dropdown(
        label="翻訳元",
        options=[
            ft.dropdown.Option("ja", "日本語"),
        ],
        value="ja",
        width=200,
    )

    target_lang_dropdown = ft.Dropdown(
        label="翻訳先",
        options=[
            ft.dropdown.Option("en", "英語"),
            ft.dropdown.Option("ko", "韓国語"),
            ft.dropdown.Option("zh", "中国語"),
        ],
        value="en",
        width=200,
    )

    input_text = ft.TextField(
        label="翻訳するテキストを入力",
        multiline=True,
        min_lines=5,
        max_lines=10,
        expand=True,
    )

    output_text = ft.TextField(
        label="翻訳結果",
        multiline=True,
        min_lines=5,
        max_lines=10,
        read_only=True,
        expand=True,
    )

    def on_translate_click(e):
        if not state["translator"]:
            return
        if not input_text.value.strip():
            return
        
        state["is_translating"] = True
        translate_btn.disabled = True
        status_text.value = "翻訳中..."
        status_text.color = ft.Colors.BLUE_700
        progress_ring.visible = True
        status_row.visible = True
        page.update()

        def run_translation():
            try:
                result = state["translator"].translate(
                    text=input_text.value,
                    source_lang=source_lang_dropdown.value,
                    target_lang=target_lang_dropdown.value,
                )
                output_text.value = result
            except Exception as ex:
                output_text.value = f"エラーが発生しました: {ex}"
            finally:
                state["is_translating"] = False
                translate_btn.disabled = False
                status_row.visible = False
                page.update()

        threading.Thread(target=run_translation, daemon=True).start()

    translate_btn = ft.Button(
        content=ft.Text("翻訳する"),
        on_click=on_translate_click,
        disabled=True,
    )

    # レイアウト
    page.add(
        ft.Container(
            content=ft.Column(
                controls=[
                    title,
                    status_row,
                    ft.Row([source_lang_dropdown, ft.Icon(ft.Icons.ARROW_FORWARD), target_lang_dropdown], alignment=ft.MainAxisAlignment.CENTER),
                    input_text,
                    ft.Row([translate_btn], alignment=ft.MainAxisAlignment.CENTER),
                    output_text,
                ],
                spacing=20,
            ),
            padding=30,
            expand=True,
        )
    )

    # モデルの非同期読み込み
    def load_model():
        try:
            model_dir = Path("translategemma-12b-ovt")
            state["translator"] = Translator(model_dir)
            status_text.value = "モデルの読み込みが完了しました"
            status_text.color = ft.Colors.GREEN_700
            progress_ring.visible = False
            translate_btn.disabled = False
            page.update()
            
            # 3秒後にステータスを隠す
            import time
            time.sleep(3)
            status_row.visible = False
            page.update()
        except Exception as ex:
            status_text.value = f"モデルの読み込みに失敗しました: {ex}"
            status_text.color = ft.Colors.RED_700
            progress_ring.visible = False
            page.update()

    threading.Thread(target=load_model, daemon=True).start()

if __name__ == "__main__":
    ft.run(main)
