# ページめくり検知システム

### <b>背景</b>

クライアントは業界に革命を起こす革新的なソリューションを開発している人工知能・コンピュータビジョン会社です。そのソリューションは小型でありながら知能を備えたデバイスに詰め込まれており、顔の認識、年齢や性別の推定、服装の種類と色の分類など日常の物体を識別し、動きを検出することもできます。

その中、モン・リーダー（Mon Reader）は視覚障害者、研究者、および大量の完全自動、高速、高品質な文書ディジタル化を必要としているあらゆる人のための、モバイル文書デジタル化ソリューションです。ユーザーはただページをめくることだけしていれば、他の全ては下記のようにモバイルアプリ形式のモン・リーダーによって処理されます。
- 低解像度のカメラプレビューからページめくりを検知
- 高解像度の写真を撮影し、文書の角を認識し、切り取る
- 取り切られた文書の歪みを補正し、鳥瞰図を取得
- テキストと背景のコントラストをより鮮明にする
- フォーマットを保ったままテキストを認識
- モン・リーダーのML技術を用いた矯正機能によって、認識された内容をさらに修正する

<img src="https://go.apziva.com/static/img/project_10_1.png"><br>

<img src="https://go.apziva.com/static/img/project_10_2.jpg"><br>

### <b>データ記述</b>

今回のプロジェクトで扱うデータはスマートフォンから収集した文書のページをめくる動画で、各動画は短く切り出されて「めくっている（flipping）」と「めくっていない（not flipping）」のラベルが付けられています。抽出されたフレームは、次の構造で名づけられて、ラベルに従って各フォルダー（すなわち、「めくっている」か「めくっていない」）の中に順番に保存されています: 動画ID_フレーム番号

フレームファイルの数（すなわち、画像データの量）はトレーニングセットとテストセットを合わせて合計約3000枚です。

### <b>目標</b>
分類モデルを開発し、各画像の中でページがめくられているかどうかを予測します。言い換えると、画像がページをめくる動作を含む一連の画像の一部（すなわちページをめくる動画の一部）であるかどうかを予測します。

### <b>成功指標</b>
F1スコアに基づいてモデルのパフォーマンスを評価。スコアが高いほど性能が優秀。

### <b>結果</b>

<u>ページめくり検知システム</u>

ページめくり動作を検出するため、バイナリ分類問題に特化した畳み込みニューラルネットワーク（CNN）を設計しました。予測能力を高めるため、以下のハイパーパラメータについてさまざまな実験と分析を行いました。
- 画像データの入力形状（input shape）；元の形状は(1920, 1080, 3)
- 各畳み込み層において：フィルタの数、カーネルのサイズ、プーリング戦略など

広範なハイパーパラメータ調整の結果、テストデータセットで非常に高い0.9832のF1スコアを達成しました。また、分類モデルが大量の高解像度画像データで学習されたといった点も、検知システムの優れた性能に大きく貢献したと考えられます。

<u>まとめ</u>

今回開発した検知システムは本番環境での実装に十分な精度を備えていますが、より多様な画像データをシステムに入れ込むことで、性能のさらなる向上が期待できます。加えて、このプロジェクトから得られた動画・画像データ処理などに関する知識とノウハウは、ページめくり以外の動きや物体なども検知できるシステムの機能拡張、新たな動体検知システムの開発に活用できると思います。

### <b>ノートブック</b>

詳細については、<a href='https://github.com/henryhyunwookim/JAPANESE-MonReader/blob/main/MonReader.ipynb'>このノートブック</a>を直接ご参照ください。

ノートブック（MonReader.ipynb）をローカル環境で実行するには、このリポジトリをクローン（複製）またはフォーク（派生）し、下記のコマンドを実行することで、必要なライブラリーをインストールしてください。

pip install -r requirements.txt

##### <i>* アプジバ（Apziva）に関連</i>